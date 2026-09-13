use smaul_native::embedding::Embedding;

#[test]
fn embedding_initialization_is_seeded() {
    let a = Embedding::new(8, 4, 123);
    let b = Embedding::new(8, 4, 123);
    assert_eq!(a.weight, b.weight);
}
