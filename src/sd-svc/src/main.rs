use std::io::Error;
use std::net::{Ipv4Addr, SocketAddr};
use tokio::net::TcpListener;

use axum::Router;
use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
  modifiers(&SecurityAddon),
  nest(
    (path = "/api/v1/sd", api = sd::SdApi)
  ),
  tags(
    (name = "SD", description = "System Dynamics API")
  )
)]
struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "api_key",
                SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("api_key"))),
            )
        }
    }
}
#[tokio::main]
async fn main() -> Result<(), Error> {
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8070));
    let listener = TcpListener::bind(&address).await?;

    let app = Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/api/v1/sd", sd::router());

    axum::serve(listener, app.into_make_service()).await
}

mod sd {
    use std::{
        collections::HashMap,
        io::{BufRead, Cursor},
        sync::Arc,
    };

    use axum::{
        extract::{Multipart, Path, State},
        response::IntoResponse,
        routing, Json, Router,
    };
    use hyper::{HeaderMap, StatusCode};
    use serde::{Deserialize, Serialize};
    use simlin_compat::{
        engine::{build_sim_with_stderrors, datamodel::Variable},
        open_xmile,
    };
    use simlin_compat::{
        engine::{datamodel::Project, SimSpecs, Vm},
        Results,
    };
    use tokio::sync::Mutex;
    use utoipa::{OpenApi, ToSchema};

    #[derive(OpenApi)]
    #[openapi(
        paths(list_projects, create_sd_project, list_models, list_model_variables, delete_sd_project, simulate_sd_project), // , search_todos, mark_done
        components(schemas(SdError, SdVariableType, SdVariable, SdSimSpecs, SdResults))
    )]
    pub(super) struct SdApi;

    /// In-memory todo store
    type Store = Mutex<Vec<Project>>;

    #[derive(Serialize, Deserialize, ToSchema)]
    #[serde(rename_all = "camelCase")]
    struct SdProject {
        project: String,
    }

    #[derive(Serialize, Deserialize, ToSchema)]
    #[serde(rename_all = "camelCase")]
    struct SdSimSpecs {
        start: f64,
        stop: f64,
        dt: f64,
        save_step: f64,
        method: String,
    }

    impl From<SimSpecs> for SdSimSpecs {
        fn from(value: SimSpecs) -> Self {
            Self {
                start: value.start,
                stop: value.stop,
                dt: value.dt,
                save_step: value.save_step,
                method: format!("{:?}", value.method),
            }
        }
    }

    #[derive(Serialize, Deserialize, ToSchema)]
    #[serde(rename_all = "camelCase")]
    struct SdResults {
        data: HashMap<String, Vec<f64>>,
        sim_specs: SdSimSpecs,
    }

    fn take_samples_from_index(data: &Box<[f64]>, i: usize, k: usize, n: usize) -> Vec<f64> {
        // Create a subslice starting from index i
        let subslice = if i < data.len() { &data[i..] } else { &[] };

        subslice
            .iter()
            .step_by(n)
            .take(k)
            .cloned() // Convert &f64 to f64
            .collect()
    }

    impl From<Results> for SdResults {
        fn from(results: Results) -> Self {
            let var_names = {
                let offset_name_map: HashMap<usize, &str> = results
                    .offsets
                    .iter()
                    .map(|(k, v)| (*v, k.as_str()))
                    .collect();
                let mut var_names: Vec<&str> = Vec::with_capacity(results.step_size);
                for i in 0..(results.step_size) {
                    let name = if offset_name_map.contains_key(&i) {
                        offset_name_map[&i]
                    } else {
                        "UNKNOWN"
                    };
                    var_names.push(name);
                }
                var_names
            };

            let data: HashMap<String, Vec<f64>> = var_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        name.to_string(),
                        take_samples_from_index(
                            &results.data,
                            i,
                            results.step_count,
                            results.step_size,
                        ),
                    )
                })
                .collect();

            Self {
                data,
                sim_specs: results.specs.into(),
            }
        }
    }

    /// SD operation errors
    #[derive(Serialize, Deserialize, ToSchema)]
    enum SdError {
        /// SD project or model already exists conflict.
        #[schema(example = "SD project or model already exists")]
        Conflict(String),
        /// SD project or model not found by id.
        #[schema(example = "id = 1")]
        NotFound(String),
        /// SD project or model operation unauthorized
        #[schema(example = "missing api key")]
        Unauthorized(String),
    }

    #[derive(Serialize, Deserialize, ToSchema)]
    enum SdVariableType {
        Stock,
        Flow,
        Aux,
        Module,
    }

    impl From<Variable> for SdVariableType {
        fn from(variable: Variable) -> Self {
            match variable {
                Variable::Stock(_) => SdVariableType::Stock,
                Variable::Flow(_) => SdVariableType::Flow,
                Variable::Aux(_) => SdVariableType::Aux,
                Variable::Module(_) => SdVariableType::Module,
            }
        }
    }

    /// SD model variables
    #[derive(Serialize, Deserialize, ToSchema)]
    #[serde(rename_all = "camelCase")]
    struct SdVariable {
        can_be_model_input: bool,
        eqn: String,
        ident: String,
        units: Option<String>,
        /// Private or public
        visibility: String,
        variable_type: SdVariableType,
    }

    impl From<Variable> for SdVariable {
        fn from(variable: Variable) -> Self {
            SdVariable {
                can_be_model_input: variable.can_be_module_input(),
                eqn: format!("{:?}", variable.get_equation()),
                ident: variable.get_ident().to_string(),
                units: variable.get_units().cloned(),
                visibility: format!("{:?}", variable.get_visibility()),
                variable_type: variable.into(),
            }
        }
    }

    pub(super) fn router() -> Router {
        let store = Arc::new(Store::default());
        Router::new()
            .route("/", routing::get(list_projects).post(create_sd_project))
            .route("/:id", routing::get(list_models).delete(delete_sd_project))
            .route("/:id/:model", routing::get(list_model_variables))
            .route("/:id/simulate", routing::get(simulate_sd_project))
            // .route("/search", routing::get(search_todos))
            // .route("/:id", routing::put(mark_done).delete(delete_todo))
            .with_state(store)
    }

    /// List all SD projects
    ///
    /// List all SD projects from in-memory storage.
    #[utoipa::path(
      get,
      path = "",
      responses(
          (status = 200, description = "List all SD projects successfully", body = [String])
      )
    )]
    async fn list_projects(State(store): State<Arc<Store>>) -> Json<Vec<String>> {
        let projects = store.lock().await.clone();

        Json(projects.iter().map(|p| p.name.clone()).collect())
    }

    // /// Todo search query
    // #[derive(Deserialize, IntoParams)]
    // struct TodoSearchQuery {
    //     /// Search by value. Search is incase sensitive.
    //     value: String,
    //     /// Search by `done` status.
    //     done: bool,
    // }

    //   /// Search Todos by query params.
    //   ///
    //   /// Search `Todo`s by query params and return matching `Todo`s.
    //   #[utoipa::path(
    //     get,
    //     path = "/search",
    //     params(
    //         TodoSearchQuery
    //     ),
    //     responses(
    //         (status = 200, description = "List matching todos by query", body = [Todo])
    //     )
    // )]
    //   async fn search_todos(
    //       State(store): State<Arc<Store>>,
    //       query: Query<TodoSearchQuery>,
    //   ) -> Json<Vec<Todo>> {
    //       Json(
    //           store
    //               .lock()
    //               .await
    //               .iter()
    //               .filter(|todo| {
    //                   todo.value.to_lowercase() == query.value.to_lowercase()
    //                       && todo.done == query.done
    //               })
    //               .cloned()
    //               .collect(),
    //       )
    //   }

    /// Register a new SD project
    ///
    /// Tries to register a new SD project to in-memory storage. In case it already exists, it will overwrite the existing one.
    ///
    /// Example
    /// curl -X POST -F 'file=@./test/i2model015-simplifiedmay24-user1.stmx' http://localhost:8070/api/v1/sd
    #[utoipa::path(
      post,
      path = "",
      request_body(content = Multipart),
      responses(
          (status = 201, description = "SD project created successfully", body = String),
          (status = 400, description = "Error creating SD Project model: invalid file or no file", body = SdError),
          (status = 422, description = "Error processing XMILE or STMX file", body = SdError),
      )
  )]
    async fn create_sd_project(
        State(store): State<Arc<Store>>,
        mut multipart: Multipart,
    ) -> impl IntoResponse {
        let mut projects = store.lock().await;

        while let Some(field) = multipart.next_field().await.unwrap() {
            let name = field.name().unwrap().to_string();
            let data: axum::body::Bytes = field.bytes().await.unwrap();
            let mut cursor = Cursor::new(data);
            let reader: &mut dyn BufRead = &mut cursor;
            let project = match open_xmile(reader) {
                Ok(project) => project,
                Err(_) => {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(format!("Error processing XMILE or STMX file: {name}")),
                    )
                        .into_response()
                }
            };
            if let Some(pos) = projects
                .iter()
                .position(|existing_project| existing_project.name == project.name)
            {
                projects.remove(pos);
            }
            projects.push(project.clone());
            return (StatusCode::CREATED, Json(project.name)).into_response();
        }
        (
            StatusCode::BAD_REQUEST,
            Json("Error creating SD Project model: invalid file or no file"),
        )
            .into_response()
    }

    //   /// Mark SD Project done by id
    //   ///
    //   /// Mark SD Project done by given id. Return only status 200 on success or 404 if Todo is not found.
    //   #[utoipa::path(
    //     put,
    //     path = "/{id}",
    //     responses(
    //         (status = 200, description = "Todo marked done successfully"),
    //         (status = 404, description = "Todo not found")
    //     ),
    //     params(
    //         ("id" = i32, Path, description = "Todo database id")
    //     ),
    //     security(
    //         (), // <-- make optional authentication
    //         ("api_key" = [])
    //     )
    // )]
    //   async fn mark_done(
    //       Path(id): Path<i32>,
    //       State(store): State<Arc<Store>>,
    //       headers: HeaderMap,
    //   ) -> StatusCode {
    //       match check_api_key(false, headers) {
    //           Ok(_) => (),
    //           Err(_) => return StatusCode::UNAUTHORIZED,
    //       }
    //       let mut todos = store.lock().await;
    //       todos
    //           .iter_mut()
    //           .find(|todo| todo.id == id)
    //           .map(|todo| {
    //               todo.done = true;
    //               StatusCode::OK
    //           })
    //           .unwrap_or(StatusCode::NOT_FOUND)
    //   }

    /// List all SD models
    ///
    /// List all SD models of an SD project.
    #[utoipa::path(
      get,
      path = "/{id}",
      responses(
        (status = 200, description = "List SD project models successfully", body = [String]),
        (status = 404, description = "SD project not found", body = SdError),
      ),
      params(
          ("id" = String, Path, description = "SD project ID")
      ),
      security(
          ("api_key" = [])
      )
    )]
    async fn list_models(
        Path(id): Path<String>,
        State(store): State<Arc<Store>>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        match check_api_key(false, headers) {
            Ok(_) => (),
            Err(error) => return error.into_response(),
        }
        let projects = store.lock().await.clone();
        if let Some(project) = projects.iter().find(|project| project.name == id) {
            let model_names: Vec<String> = project.models.iter().map(|m| m.name.clone()).collect();
            (StatusCode::OK, Json(model_names)).into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(SdError::NotFound(format!("id = {id}"))),
            )
                .into_response()
        }
    }

    /// List all variables of SD model
    ///
    /// List all variables in an SD model of an SD project.
    #[utoipa::path(
      get,
      path = "/{id}/{model}",
      responses(
        (status = 200, description = "List SD project model variables successfully", body = [String]),
        (status = 404, description = "SD project or model not found", body = SdError),
      ),
      params(
        ("id" = String, Path, description = "SD project ID"),
        ("model" = String, Path, description = "SD model ID"),
      ),
      security(
          ("api_key" = [])
      )
    )]
    async fn list_model_variables(
        Path((id, model_id)): Path<(String, String)>,
        State(store): State<Arc<Store>>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        match check_api_key(false, headers) {
            Ok(_) => (),
            Err(error) => return error.into_response(),
        }
        let projects = store.lock().await.clone();
        if let Some(project) = projects.iter().find(|project| project.name == id) {
            if let Some(model) = project.models.iter().find(|model| model.name == model_id) {
                let model_variables: Vec<SdVariable> =
                    model.variables.iter().map(|v| v.clone().into()).collect();
                (StatusCode::OK, Json(model_variables)).into_response()
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(SdError::NotFound(format!("model = {model_id}"))),
                )
                    .into_response()
            }
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(SdError::NotFound(format!("id = {id}"))),
            )
                .into_response()
        }
    }

    fn simulate(project: &Project) -> Results {
        let sim = build_sim_with_stderrors(project).unwrap();
        let compiled = sim.compile().unwrap();
        let mut vm = Vm::new(compiled).unwrap();
        vm.run_to_end().unwrap();
        vm.into_results()
    }

    /// Simulate SD project
    ///
    /// Run a simulation of an SD project.
    #[utoipa::path(
      get,
      path = "/{id}/simulate",
      responses(
        (status = 200, description = "Simulation executed successfully", body = SdResults),
        (status = 404, description = "SD project not found", body = SdError),
      ),
      params(
          ("id" = String, Path, description = "SD database id")
      ),
      security(
          ("api_key" = [])
      )
    )]
    async fn simulate_sd_project(
        Path(id): Path<String>,
        State(store): State<Arc<Store>>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        match check_api_key(false, headers) {
            Ok(_) => (),
            Err(error) => return error.into_response(),
        }
        let projects = store.lock().await.clone();
        if let Some(project) = projects.iter().find(|project| project.name == id) {
            let results = simulate(&project);
            // println!("{:?}", results.offsets);
            let sd_results: SdResults = results.into();
            (StatusCode::OK, Json(sd_results)).into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(SdError::NotFound(format!("id = {id}"))),
            )
                .into_response()
        }
    }

    /// Delete SD Project by id
    ///
    /// Delete SD Project from in-memory storage by id. Returns either 200 success of 404 with SdError if SD Project is not found.
    #[utoipa::path(
      delete,
      path = "/{id}",
      responses(
          // (status = 401, description = "Unauthorized to delete Todo", body = SdError, example = json!(SdError::Unauthorized(String::from("missing api key")))),
          (status = 404, description = "SD Project not found", body = SdError, example = json!(SdError::NotFound(String::from("id = 1"))))
      ),
      params(
          ("id" = String, Path, description = "SD database id")
      ),
      security(
          ("api_key" = [])
      )
    )]
    async fn delete_sd_project(
        Path(id): Path<String>,
        State(store): State<Arc<Store>>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        match check_api_key(false, headers) {
            Ok(_) => (),
            Err(error) => return error.into_response(),
        }

        let mut projects = store.lock().await;

        let len = projects.len();

        projects.retain(|project| project.name != id);

        if projects.len() != len {
            StatusCode::OK.into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(SdError::NotFound(format!("id = {id}"))),
            )
                .into_response()
        }
    }

    // normally you should create a middleware for this but this is sufficient for sake of example.
    fn check_api_key(
        require_api_key: bool,
        headers: HeaderMap,
    ) -> Result<(), (StatusCode, Json<SdError>)> {
        match headers.get("api_key") {
            // Some(header) if header != "utoipa-rocks" => Err((
            //     StatusCode::UNAUTHORIZED,
            //     Json(SdError::Unauthorized(String::from("incorrect api key"))),
            // )),
            None if require_api_key => Err((
                StatusCode::UNAUTHORIZED,
                Json(SdError::Unauthorized(String::from("missing api key"))),
            )),
            _ => Ok(()),
        }
    }
}
