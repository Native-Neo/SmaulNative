use smaul_native::embedding::Embedding;

#[test]
fn embedding_parameter_count_matches_storage() {
    let embedding = Embedding::new(10, 6, 9);
    assert_eq!(embedding.parameter_count(), 60);
    assert_eq!(embedding.flatten_weight().len(), 60);
}
