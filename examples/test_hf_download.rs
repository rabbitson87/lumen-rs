fn main() {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::Repo;
    
    println!("Testing HF Hub download...");
    let api = match ApiBuilder::new().build() {
        Ok(a) => a,
        Err(e) => { println!("API build error: {e:?}"); return; }
    };
    let repo = api.repo(Repo::model("google/gemma-4-E4B-it".to_string()));
    match repo.get("config.json") {
        Ok(path) => println!("OK: {}", path.display()),
        Err(e) => println!("ERR: {e:?}"),
    }
}
