


use std::sync::Mutex;
use simple_neural_network_project::{back, backprop, neural_network::{self, NeuralNetwork, TypeNeuronValue}, traning_handeler::TraningHandeler};
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
            .route("/pkg/simple_neural_network_project.js", web::get().to(js))
            .route("/pkg/simple_neural_network_project_bg.wasm", web::get().to(bg))
            .route("/dataset_1.json", web::get().to(dataset_1))
            .route("/style.css", web::get().to(style))
            .service(Files::new("/", "./pkg/"))
            //.service(echo)
            //.route("/foo.csv", web::get().to(csv))
            /* 
            .service(
                web::scope("/app")
                .route("/index.html", web::get().to(hello))
            )
            */
            /* 
            .service(
                // prefixes all resources and routes attached to it...
                web::scope("/app")
                    // ...so this handles requests for `GET /app/index.html`
                    .route("/hello", web::get().to(index))
                    .route("/css_style.css", web::get().to(style))
                    .route("/destination", web::get().to(destination))
                    .route("/v", web::get().to(d3))
            )
            */
            /*
            .service(
                // prefixes all resources and routes attached to it...
                web::scope("/app")
                    // ...so this handles requests for `GET /app/index.html`
                    .route("/css", web::get().to(index)),
            )
            */
            //.route("/time", web::get().to(|| async { "Current time: ...".to_string() }))
            //.service(time)
            //.route("/h.html", web::get().to(manual_hello))
            //.service(Files::new("/", "./static"))

    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}



async fn html() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/visualize.html").await.unwrap()
}

async fn style() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/style.css").await.unwrap()
}

async fn dataset_1() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/dataset_1.json").await.unwrap()
}

async fn js() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/pkg/simple_neural_network_project.js").await.unwrap()
}

async fn bg() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/pkg/simple_neural_network_project_bg.wasm").await.unwrap()
}