


use std::sync::Mutex;
use simple_neural_network_project::{backprop, neural_network::{self, NeuralNetwork, TypeNeuronValue}, traning_handeler::TraningHandeler};
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_files::{Files, NamedFile};
use serde::Serialize;
use csv::*;


// ideas for vizualisation
//https://www.chartjs.org/docs/latest/samples/other-charts/bubble.html
// chart.js
//

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/visualize", web::get().to(html))
            .route("/jquery-csv.js", web::get().to(csv))
            .route("/pkg/simple_neural_network_project.js", web::get().to(js))
            .route("/pkg/simple_neural_network_project_bg.wasm", web::get().to(bg))
            .route("/dataset_1.csv", web::get().to(dataset_1))
            .route("/dataset_2.csv", web::get().to(dataset_2))
            .route("/style.css", web::get().to(style))
            .service(Files::new("/", "./pkg/"))

    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}



async fn html() -> impl Responder {
    NamedFile::open_async("./src/bin/web/visualize.html").await.unwrap()
}

async fn style() -> impl Responder {
    NamedFile::open_async("./src/bin/web/style.css").await.unwrap()
}

async fn dataset_1() -> impl Responder {
    NamedFile::open_async("./src/bin/web/dataset_1.csv").await.unwrap()
}

async fn dataset_2() -> impl Responder {
    NamedFile::open_async("./src/bin/web/dataset_2.csv").await.unwrap()
}

async fn csv() -> impl Responder {
    NamedFile::open_async("./src/bin/web/jquery-csv.js").await.unwrap()
}

async fn js() -> impl Responder {
    NamedFile::open_async("./pkg/simple_neural_network_project.js").await.unwrap()
}

async fn bg() -> impl Responder {
    NamedFile::open_async("./pkg/simple_neural_network_project_bg.wasm").await.unwrap()
}