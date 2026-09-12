use std::collections::HashMap;

pub(crate) fn preprocess_isle(input: &str) -> Result<String, String> {
    let lines: Vec<(usize, String)> = input
        .lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.to_string()))
        .collect();
    let (nodes, idx) = parse_preprocess_nodes(&lines, 0, false)?;
    debug_assert_eq!(idx, lines.len());

    let mut out = String::new();
    expand_preprocess_nodes(&nodes, &HashMap::new(), &mut out)?;
    Ok(out)
}

#[derive(Debug, Clone)]
enum PreprocessNode {
    Text { text: String },
    For(PreprocessFor),
}

#[derive(Debug, Clone)]
struct PreprocessFor {
    names: Vec<String>,
    tuples: Vec<Vec<String>>,
    body: Vec<PreprocessNode>,
}

fn parse_preprocess_nodes(
    lines: &[(usize, String)],
    mut idx: usize,
    stop_at_end: bool,
) -> Result<(Vec<PreprocessNode>, usize), String> {
    let mut nodes = Vec::new();

    while idx < lines.len() {
        let (line_no, line) = &lines[idx];
        let trimmed = line.trim_start();

        if trimmed == "end" {
            if stop_at_end {
                return Ok((nodes, idx + 1));
            }
            return Err(format!("unexpected `end` on line {}", line_no));
        }

        if let Some(spec) = trimmed.strip_prefix("for ") {
            let (node, next_idx) = parse_for_block(lines, idx, spec)?;
            nodes.push(node);
            idx = next_idx;
            continue;
        }

        nodes.push(PreprocessNode::Text { text: line.clone() });
        idx += 1;
    }

    if stop_at_end {
        Err("missing `end` for `for` block".to_string())
    } else {
        Ok((nodes, idx))
    }
}

fn parse_for_block(
    lines: &[(usize, String)],
    idx: usize,
    spec: &str,
) -> Result<(PreprocessNode, usize), String> {
    let (header_line, _) = &lines[idx];
    let (names, inline_values) = parse_for_header(spec, *header_line)?;

    if let Some(tuples) = inline_values {
        let (body, next_idx) = parse_preprocess_nodes(lines, idx + 1, true)
            .map_err(|err| format!("{err} (starting on line {})", header_line))?;
        return Ok((
            PreprocessNode::For(PreprocessFor {
                names,
                tuples,
                body,
            }),
            next_idx,
        ));
    }

    let do_idx = find_for_do(lines, idx + 1, *header_line)?;
    let tuples = parse_for_cases(&lines[idx + 1..do_idx], names.len(), *header_line)?;
    let (body, next_idx) = parse_preprocess_nodes(lines, do_idx + 1, true)
        .map_err(|err| format!("{err} (starting on line {})", header_line))?;

    Ok((
        PreprocessNode::For(PreprocessFor {
            names,
            tuples,
            body,
        }),
        next_idx,
    ))
}

fn expand_preprocess_nodes(
    nodes: &[PreprocessNode],
    vars: &HashMap<String, String>,
    out: &mut String,
) -> Result<(), String> {
    for node in nodes {
        match node {
            PreprocessNode::Text { text } => {
                out.push_str(&substitute_vars(text, vars));
                out.push('\n');
            }
            PreprocessNode::For(for_node) => {
                for tuple in &for_node.tuples {
                    let mut scoped = vars.clone();
                    for (name, value) in for_node.names.iter().zip(tuple.iter()) {
                        scoped.insert(name.clone(), value.clone());
                    }

                    let mut expanded = String::new();
                    expand_preprocess_nodes(&for_node.body, &scoped, &mut expanded)?;
                    out.push_str(&expanded);
                    if !expanded.ends_with('\n') {
                        out.push('\n');
                    }
                }
            }
        }
    }

    Ok(())
}

fn parse_for_names(spec: &str, line_no: usize) -> Result<Vec<String>, String> {
    let names: Vec<String> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if names.is_empty() {
        Err(format!(
            "invalid `for` header on line {}: missing loop variables in {spec}",
            line_no
        ))
    } else {
        Ok(names)
    }
}

fn parse_for_header(
    spec: &str,
    line_no: usize,
) -> Result<(Vec<String>, Option<Vec<Vec<String>>>), String> {
    let Some((lhs, rhs)) = spec.split_once(" in") else {
        return Err(format!(
            "invalid `for` header on line {}: expected `for (vars) in`",
            line_no
        ));
    };

    let lhs = lhs.trim();
    let Some(inner) = lhs.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
        return Err(format!(
            "invalid `for` header on line {}: loop variables must be wrapped in parentheses",
            line_no
        ));
    };

    let names = parse_for_names(inner, line_no)?;

    let rhs = rhs.trim();
    if rhs.is_empty() {
        return Ok((names, None));
    }

    let values = rhs
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|tuple| parse_for_tuple(tuple, names.len(), line_no))
        .collect::<Result<Vec<_>, _>>()?;

    if values.is_empty() {
        return Err(format!(
            "invalid `for` header on line {}: no tuples found in {spec}",
            line_no
        ));
    }

    Ok((names, Some(values)))
}

fn parse_for_cases(
    lines: &[(usize, String)],
    arity: usize,
    header_line: usize,
) -> Result<Vec<Vec<String>>, String> {
    let mut values = Vec::new();
    for (line_no, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(";;") {
            continue;
        }
        values.push(parse_for_tuple(trimmed, arity, *line_no)?);
    }
    if values.is_empty() {
        return Err(format!(
            "invalid `for` block on line {}: no cases found before `do`",
            header_line
        ));
    }
    Ok(values)
}

fn parse_for_tuple(tuple: &str, arity: usize, line_no: usize) -> Result<Vec<String>, String> {
    let vals: Vec<String> = tuple
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if vals.len() != arity {
        Err(format!(
            "invalid `for` tuple on line {} `{tuple}`: expected {} values, found {}",
            line_no,
            arity,
            vals.len()
        ))
    } else {
        Ok(vals)
    }
}

fn find_for_do(
    lines: &[(usize, String)],
    mut idx: usize,
    header_line: usize,
) -> Result<usize, String> {
    let mut depth = 0usize;
    while idx < lines.len() {
        let (_, line) = &lines[idx];
        let trimmed = line.trim_start();
        if trimmed.strip_prefix("for ").is_some() {
            depth += 1;
        } else if trimmed == "end" {
            if depth == 0 {
                return Err(format!(
                    "missing `do` for `for` block starting on line {}",
                    header_line
                ));
            }
            depth -= 1;
        } else if trimmed == "do" && depth == 0 {
            return Ok(idx);
        }
        idx += 1;
    }
    Err(format!(
        "missing `do` for `for` block starting on line {}",
        header_line
    ))
}

fn substitute_vars(text: &str, vars: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + 2..];
        let Some(end) = after_open.find("}}") else {
            out.push_str(&rest[start..]);
            return out;
        };

        let placeholder = &after_open[..end];
        let key = placeholder.trim();
        if let Some(value) = vars.get(key) {
            out.push_str(value);
        } else {
            out.push_str("{{");
            out.push_str(placeholder);
            out.push_str("}}");
        }
        rest = &after_open[end + 2..];
    }

    out.push_str(rest);
    out
}
