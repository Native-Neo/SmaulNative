pub fn next_token_targets(tokens: &[usize]) -> Vec<usize> {
    if tokens.len() < 2 { return Vec::new(); }
    tokens[1..].to_vec()
}

pub fn input_tokens(tokens: &[usize]) -> Vec<usize> {
    if tokens.len() < 2 { return Vec::new(); }
    tokens[..tokens.len() - 1].to_vec()
}
