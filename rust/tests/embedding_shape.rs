use smaul_native::embedding::Embedding;

#[test]
fn embedding_preserves_token_sequence_shape() {
    let embedding = Embedding::new(16, 8, 7);
    assert_eq!(embedding.forward(&[0, 3, 15]).dim(), (3, 8));
}
