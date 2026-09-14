use serde_json::Value;

fn normalize_text(text: &str) -> String {
    text.nfkc().collect::<String>().replace("\r\n", "\n").replace('\r', "\n").chars().filter(|c| !c.is_control() || *c == '\n' || *c == '\t').collect::<String>().trim().to_owned()
}

pub fn filter_text(text: &str, dataset: &str, min_chars: usize, max_chars: usize) -> Option<String> {
    let text = normalize_text(text);
    if text.len() < min_chars || text.len() > max_chars || text.contains('\u{fffd}') { return None; }
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.chars().collect::<std::collections::HashSet<_>>().len() < 8 { return None; }
    let url_only = text.strip_prefix("http://").or_else(|| text.strip_prefix("https://")).or_else(|| text.strip_prefix("www."));
    if url_only.is_some() && !text.chars().any(char::is_whitespace) { return None; }
    let letters = text.chars().filter(|c| c.is_alphabetic()).count().max(1) as f32;
    if dataset == "hindi" {
        let d = text.chars().filter(|c| matches!(*c, '\u{0900}'..='\u{097f}')).count();
        if d < 8 || d as f32 / letters < 0.20 { return None; }
    } else if dataset == "english" {
        let l = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
        if l < 12 || l as f32 / letters < 0.50 { return None; }
    }
    Some(text)
}

pub fn filter_record(record: &Value, dataset: &str, min_chars: usize, max_chars: usize) -> Option<String> {
    let object = record.as_object()?;
    let lower = object.iter().map(|(k, v)| (k.to_ascii_lowercase(), v)).collect::<std::collections::HashMap<_, _>>();
    let prompt = lower.get("prompt").and_then(|v| v.as_str());
    let completion = lower.get("completion").and_then(|v| v.as_str());
    let text = match (prompt, completion) {
        (Some(a), Some(b)) => format!("{a}\n{b}"),
        (Some(a), None) => a.to_owned(),
        (None, Some(b)) => b.to_owned(),
        _ => lower.get("text").and_then(|v| v.as_str())?.to_owned(),
    };
    filter_text(&text, dataset, min_chars, max_chars)
}

pub fn filter_jsonl(input: impl AsRef<std::path::Path>, output: impl AsRef<std::path::Path>, dataset: &str, min_chars: usize, max_chars: usize) -> Result<usize, String> {
    use std::io::{BufRead, Write};
    let reader = std::io::BufReader::new(std::fs::File::open(input).map_err(|e| e.to_string())?);
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output).map_err(|e| e.to_string())?);
    let mut kept = 0;
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let Ok(record) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(text) = filter_record(&record, dataset, min_chars, max_chars) {
            writeln!(writer, "{}", serde_json::json!({"text": text})).map_err(|e| e.to_string())?;
            kept += 1;
        }
    }
    Ok(kept)
}

use unicode_normalization::UnicodeNormalization;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filters_language() {
        assert!(filter_text("This is a sufficiently long English training document with many letters.", "english", 20, 1000).is_some());
        assert!(filter_text("short", "english", 20, 1000).is_none());
        assert!(filter_text("https://example.com", "auto", 5, 1000).is_none());
    }
    #[test]
    fn normalizes_and_removes_controls() {
        assert_eq!(filter_text("  A\r\nB\u{0001}  ", "auto", 1, 100), Some("A\nB".into()));
    }
}
