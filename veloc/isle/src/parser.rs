#![allow(unused_assignments)]

use crate::ast::*;
use crate::lexer::Token;
use logos::{Logos, SpannedIter};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

fn token_to_string(t: &Token) -> String {
    match t {
        Token::Ident(s) => s.clone(),
        Token::DefMacro => "def-macro".to_string(),
        Token::DefTemplate => "def-template".to_string(),
        Token::DefPseudoInst => "def-pseudo-inst".to_string(),
        Token::DefInst => "def-inst".to_string(),
        Token::RewriteRule => "rewrite-rule".to_string(),
        Token::Operands => "operands".to_string(),
        Token::ImplicitUses => "implicit-uses".to_string(),
        Token::ImplicitDefs => "implicit-defs".to_string(),
        Token::Clobbers => "clobbers".to_string(),
        Token::Encode => "encode".to_string(),
        Token::Byte => "byte".to_string(),
        Token::Rel32 => "rel32".to_string(),
        Token::If => "if".to_string(),
        Token::Else => "else".to_string(),
        Token::Use => "use".to_string(),
        Token::Def => "def".to_string(),
        Token::Fixed => "fixed".to_string(),
        Token::Tied => "tied".to_string(),
        Token::Imm => "imm".to_string(),
        Token::Block => "block".to_string(),
        Token::Global => "global".to_string(),
        Token::Type => "type".to_string(),
        Token::Primitive => "primitive".to_string(),
        Token::DefFeature => "def-feature".to_string(),
        Token::DefCpu => "def-cpu".to_string(),
        Token::DefAbi => "def-abi".to_string(),
        Token::Features => "features".to_string(),
        Token::Limitations => "limitations".to_string(),
        Token::RegisterClass => "register-class".to_string(),
        Token::DefReg => "def-reg".to_string(),
        Token::DefRegClass => "def-regclass".to_string(),
        Token::Size => "size".to_string(),
        Token::Alias => "alias".to_string(),
        Token::Class => "class".to_string(),
        Token::HwEnc => "hw-enc".to_string(),
        Token::SelectRule => "select-rule".to_string(),
        Token::CombineRule => "combine-rule".to_string(),
        Token::PeepholeRule => "peephole-rule".to_string(),
        Token::DefExtractor => "def-extractor".to_string(),
        Token::Template => "template".to_string(),
        Token::And => "and".to_string(),
        Token::RootKw => "root".to_string(),
        Token::MatchKw => "match".to_string(),
        Token::MatchPairKw => "match-pair".to_string(),
        Token::WhenKw => "when".to_string(),
        Token::ReplaceKw => "replace".to_string(),
        Token::EmitKw => "emit".to_string(),
        Token::CoversKw => "covers".to_string(),
        Token::CostKw => "cost".to_string(),
        Token::PriorityKw => "priority".to_string(),
        _ => format!("{:?}", t),
    }
}

#[derive(Error, Diagnostic, Debug)]
pub enum ParseError {
    #[error("Unexpected token: expected {expected}, found {found}")]
    #[diagnostic(
        code(isle::unexpected_token),
        help("Try checking the syntax or keywords here.")
    )]
    UnexpectedToken {
        #[label("expected {expected}")]
        span: SourceSpan,
        expected: String,
        found: String,
    },
    #[error("Unexpected end of input")]
    #[diagnostic(code(isle::eof))]
    UnexpectedEof,
    #[error("Built-in function {name} expected {expected} arguments, found {found}")]
    #[diagnostic(code(isle::arg_count))]
    ArgumentCountError {
        #[label("{name} expects {expected} args")]
        span: SourceSpan,
        name: String,
        expected: usize,
        found: usize,
    },
    #[error("Invalid argument for {name}: {message}")]
    #[diagnostic(code(isle::invalid_arg))]
    InvalidArgError {
        #[label("{message}")]
        span: SourceSpan,
        name: String,
        message: String,
    },
    #[error("Missing required field: {field}")]
    #[diagnostic(code(isle::missing_field))]
    MissingField {
        #[label("missing field {field}")]
        span: SourceSpan,
        field: String,
    },
    #[error("Lexer error")]
    #[diagnostic(code(isle::lexer_error))]
    LexError {
        #[label("invalid token")]
        span: SourceSpan,
    },
}

pub struct Parser<'a> {
    lexer: SpannedIter<'a, Token>,
    peeked: Option<(Result<Token, ()>, std::ops::Range<usize>)>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Token::lexer(input).spanned(),
            peeked: None,
        }
    }

    fn peek(&mut self) -> Result<Option<(Token, std::ops::Range<usize>)>, ParseError> {
        if self.peeked.is_none() {
            self.peeked = self.lexer.next();
        }
        match &self.peeked {
            Some((Ok(t), span)) => Ok(Some((t.clone(), span.clone()))),
            Some((Err(_), span)) => Err(ParseError::LexError {
                span: span.clone().into(),
            }),
            None => Ok(None),
        }
    }

    fn next(&mut self) -> Result<Option<(Token, std::ops::Range<usize>)>, ParseError> {
        let res = self.peek()?;
        self.peeked = None;
        Ok(res)
    }

    fn err_unexpected<T>(
        &self,
        span: std::ops::Range<usize>,
        expected: &str,
        found: &Token,
    ) -> Result<T, ParseError> {
        Err(ParseError::UnexpectedToken {
            span: span.into(),
            expected: expected.to_string(),
            found: token_to_string(found),
        })
    }

    fn expect(&mut self, expected: Token) -> Result<std::ops::Range<usize>, ParseError> {
        match self.next()? {
            Some((t, span)) if t == expected => Ok(span),
            Some((t, span)) => self.err_unexpected(span, &format!("{:?}", expected), &t),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.next()? {
            Some((Token::Ident(s), _)) => Ok(s),
            Some((Token::Imm, _)) => Ok("imm".to_string()),
            Some((Token::Template, _)) => Ok("template".to_string()),
            Some((Token::If, _)) => Ok("if".to_string()),
            Some((Token::Else, _)) => Ok("else".to_string()),
            Some((Token::HwEnc, _)) => Ok("hw-enc".to_string()),
            Some((Token::Class, _)) => Ok("class".to_string()),
            Some((Token::Block, _)) => Ok("block".to_string()),
            Some((Token::Global, _)) => Ok("global".to_string()),
            Some((Token::Rel32, _)) => Ok("rel32".to_string()),
            Some((t, span)) => self.err_unexpected(span, "Identifier", &t),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn expect_node_name(&mut self) -> Result<String, ParseError> {
        if let Some((Token::At, _)) = self.peek()? {
            self.next()?;
        }
        self.expect_ident()
    }

    pub fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut defs = Vec::new();
        while self.peek()?.is_some() {
            defs.push(self.parse_def()?);
        }
        Ok(Module { defs })
    }

    fn parse_def(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::LParen)?;
        let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        let def = match tok {
            Token::DefMacro => self.parse_def_macro()?,
            Token::DefTemplate => self.parse_def_template()?,
            Token::DefInst => self.parse_def_inst()?,
            Token::DefPseudoInst => self.parse_def_pseudo_inst()?,
            Token::Type => self.parse_def_type()?,
            Token::DefReg => self.parse_def_reg()?,
            Token::DefRegClass => self.parse_def_reg_class()?,
            Token::DefExtractor => self.parse_def_extractor()?,
            Token::DefFeature => self.parse_def_feature()?,
            Token::DefCpu => self.parse_def_cpu()?,
            Token::DefAbi => self.parse_def_abi()?,
            Token::Decl => self.parse_def_decl()?,
            Token::SelectRule => self.parse_select_rule()?,
            Token::RewriteRule => self.parse_rewrite_rule()?,
            Token::CombineRule => self.parse_combine_rule()?,
            Token::PeepholeRule => self.parse_peephole_rule()?,
            _ => {
                return self.err_unexpected(
                    span,
                    "def-macro, def-template, def-inst, def-pseudo-inst, type, def-reg, def-extractor, def-feature, def-cpu, def-abi, decl, select-rule, rewrite-rule, combine-rule, or peephole-rule",
                    &tok,
                );
            }
        };
        self.expect(Token::RParen)?;
        Ok(def)
    }

    fn parse_def_type(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::Type)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        let kind = match tok {
            Token::Primitive => {
                self.next()?;
                TypeKind::Primitive(self.expect_ident()?)
            }
            Token::RegisterClass => {
                self.next()?;
                TypeKind::RegisterClass(self.expect_ident()?)
            }
            _ => return self.err_unexpected(span, "primitive or register-class", &tok),
        };
        self.expect(Token::RParen)?;
        Ok(Def::Type(TypeDef { name, kind }))
    }

    fn parse_def_reg(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefReg)?;
        let name = self.expect_ident()?;

        let mut size = 0;
        let mut alias = None;
        let mut hw_enc = 0;
        let mut reserved = false;
        let mut roles = Vec::new();

        while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::Size => {
                    self.next()?;
                    size = match self.next()? {
                        Some((Token::Int(i), _)) => i as u32,
                        Some((t, span)) => {
                            return self.err_unexpected(span, "Integer for size", &t);
                        }
                        None => return Err(ParseError::UnexpectedEof),
                    };
                }
                Token::Alias => {
                    self.next()?;
                    let parent = self.expect_ident()?;
                    let range = match self.next()? {
                        Some((Token::Range(r), _)) => r,
                        Some((t, span)) => {
                            return self.err_unexpected(span, "Range (e.g. 0-31)", &t);
                        }
                        None => return Err(ParseError::UnexpectedEof),
                    };
                    alias = Some((parent, range));
                }
                Token::HwEnc => {
                    self.next()?;
                    hw_enc = match self.next()? {
                        Some((Token::Int(i), _)) => i as u32,
                        Some((t, span)) => {
                            return self.err_unexpected(span, "Integer for hw-enc", &t);
                        }
                        None => return Err(ParseError::UnexpectedEof),
                    };
                }
                Token::Reserved => {
                    self.next()?;
                    reserved = true;
                }
                Token::Role => {
                    self.next()?;
                    roles.push(self.expect_ident()?);
                }
                _ => {
                    return self.err_unexpected(
                        span,
                        "size, alias, hw-enc, reserved, or role",
                        &tok,
                    )
                }
            }
            self.expect(Token::RParen)?;
        }

        Ok(Def::Reg(RegDef {
            name,
            size,
            alias,
            hw_enc,
            reserved,
            roles,
        }))
    }

    fn parse_def_reg_class(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefRegClass)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let mut regs = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            regs.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;
        Ok(Def::RegClass(RegClassDef { name, regs }))
    }

    fn parse_def_feature(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefFeature)?;
        let name = self.expect_ident()?;
        let doc = match self.next()? {
            Some((Token::String(s), _)) => s,
            Some((t, span)) => {
                return self.err_unexpected(span, "string literal for documentation", &t);
            }
            None => return Err(ParseError::UnexpectedEof),
        };
        Ok(Def::Feature(FeatureDef { name, doc }))
    }

    fn parse_def_cpu(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefCpu)?;
        let name = match self.next()? {
            Some((Token::String(s), _)) => s,
            Some((t, span)) => return self.err_unexpected(span, "string literal for CPU name", &t),
            None => return Err(ParseError::UnexpectedEof),
        };

        let mut features = Vec::new();
        let mut limitations = Vec::new();

        while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::Features => {
                    self.next()?;
                    while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                        features.push(self.expect_ident()?);
                    }
                }
                Token::Limitations => {
                    self.next()?;
                    while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                        limitations.push(self.expect_ident()?);
                    }
                }
                _ => return self.err_unexpected(span, "features or limitations", &tok),
            }
            self.expect(Token::RParen)?;
        }

        Ok(Def::Cpu(CpuDef {
            name,
            features,
            limitations,
        }))
    }

    fn parse_def_abi(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefAbi)?;
        let name = self.expect_ident()?;

        let mut arch = None;
        let mut stack = AbiStackDef::default();
        let mut args = Vec::new();
        let mut returns = Vec::new();
        let mut preserved = Vec::new();
        let mut classifier = None;
        let abi_span = self.peek()?.map(|(_, span)| span).unwrap_or(0..0);

        while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
            self.expect(Token::LParen)?;
            let field = self.expect_ident()?;
            match field.as_str() {
                "arch" => {
                    arch = Some(self.expect_ident()?);
                }
                "stack" => {
                    while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
                        self.expect(Token::LParen)?;
                        let stack_field = self.expect_ident()?;
                        match stack_field.as_str() {
                            "align" => {
                                stack.align = Some(self.expect_int()? as u32);
                            }
                            "incoming-base" => {
                                let reg = self.expect_ident()?;
                                let offset = self.expect_int()? as i32;
                                stack.incoming_base = Some((reg, offset));
                            }
                            "outgoing-slot" => {
                                let size = self.expect_int()? as u32;
                                let align = self.expect_int()? as u32;
                                stack.outgoing_slot = Some((size, align));
                            }
                            _ => {
                                return Err(ParseError::InvalidArgError {
                                    span: abi_span.clone().into(),
                                    name: "def-abi".to_string(),
                                    message: format!("unsupported stack field `{}`", stack_field),
                                });
                            }
                        }
                        self.expect(Token::RParen)?;
                    }
                }
                "args" => {
                    args = self.parse_abi_class_regs_list()?;
                }
                "returns" => {
                    returns = self.parse_abi_class_regs_list()?;
                }
                "preserved" => {
                    preserved = self.parse_abi_preserved_sets()?;
                }
                "classifier" => {
                    classifier = Some(self.expect_ident()?);
                }
                _ => {
                    return Err(ParseError::InvalidArgError {
                        span: abi_span.clone().into(),
                        name: "def-abi".to_string(),
                        message: format!("unsupported abi field `{}`", field),
                    });
                }
            }
            self.expect(Token::RParen)?;
        }

        Ok(Def::Abi(AbiDef {
            name,
            arch: arch.ok_or(ParseError::MissingField {
                span: abi_span.into(),
                field: "(arch ...)".to_string(),
            })?,
            stack,
            args,
            returns,
            preserved,
            classifier,
        }))
    }

    fn parse_abi_class_regs_list(&mut self) -> Result<Vec<AbiClassRegsDef>, ParseError> {
        let mut entries = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
            self.expect(Token::LParen)?;
            let kind = self.expect_ident()?;
            if kind != "class" {
                return Err(ParseError::InvalidArgError {
                    span: self.peek()?.map(|(_, span)| span).unwrap_or(0..0).into(),
                    name: "def-abi".to_string(),
                    message: format!("expected `(class ...)`, found `{}`", kind),
                });
            }

            let class = self.expect_ident()?;
            self.expect(Token::LParen)?;
            let regs_kw = self.expect_ident()?;
            if regs_kw != "regs" {
                return Err(ParseError::InvalidArgError {
                    span: self.peek()?.map(|(_, span)| span).unwrap_or(0..0).into(),
                    name: "def-abi".to_string(),
                    message: format!("expected `(regs ...)`, found `{}`", regs_kw),
                });
            }
            let mut regs = Vec::new();
            while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                regs.push(self.expect_ident()?);
            }
            self.expect(Token::RParen)?;
            self.expect(Token::RParen)?;
            entries.push(AbiClassRegsDef { class, regs });
        }
        Ok(entries)
    }

    fn parse_abi_preserved_sets(&mut self) -> Result<Vec<AbiPreservedSetDef>, ParseError> {
        let mut sets = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
            self.expect(Token::LParen)?;
            let bank = self.expect_ident()?;
            let mut regs = Vec::new();
            while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                regs.push(self.expect_ident()?);
            }
            self.expect(Token::RParen)?;
            sets.push(AbiPreservedSetDef { bank, regs });
        }
        Ok(sets)
    }

    fn parse_def_decl(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::Decl)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            params.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;
        Ok(Def::Decl(DeclDef { name, params }))
    }

    fn parse_select_rule(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::SelectRule)?;

        let mut attrs = RuleAttrs::default();
        let mut patterns = Vec::new();
        let mut emit = None;
        let rule_span = self.peek()?.map(|(_, span)| span).unwrap_or(0..0);

        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::RootKw => {
                    self.next()?;
                    attrs.root = Some(self.expect_node_name()?);
                }
                Token::MatchKw => {
                    self.next()?;
                    patterns = self.parse_rule_patterns_until_rparen()?;
                }
                Token::WhenKw => {
                    self.next()?;
                    attrs.predicates = self.parse_predicate_list()?;
                }
                Token::EmitKw => {
                    self.next()?;
                    emit = Some(self.parse_constructor()?);
                }
                Token::CoversKw => {
                    self.next()?;
                    attrs.covers = self.parse_cover_list()?;
                }
                Token::CostKw => {
                    self.next()?;
                    attrs.cost = Some(self.expect_int()?);
                }
                Token::PriorityKw => {
                    self.next()?;
                    attrs.priority = Some(self.expect_int()?);
                }
                _ => return self.err_unexpected(span, "valid select-rule field", &tok),
            }
            self.expect(Token::RParen)?;
        }

        if patterns.is_empty() {
            return Err(ParseError::MissingField {
                span: rule_span.into(),
                field: "(match ...)".to_string(),
            });
        }
        if patterns.len() != 1 {
            return Err(ParseError::InvalidArgError {
                span: rule_span.into(),
                name: "select-rule".to_string(),
                message: "select-rule currently expects exactly one (match ...) pattern"
                    .to_string(),
            });
        }

        let emit = emit.ok_or(ParseError::MissingField {
            span: rule_span.into(),
            field: "(emit ...)".to_string(),
        })?;

        Ok(Def::SelectRule(SelectRuleDef {
            attrs,
            patterns,
            emit,
        }))
    }

    fn parse_rewrite_rule(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::RewriteRule)?;

        let mut attrs = RuleAttrs::default();
        let mut patterns = Vec::new();
        let mut replace = None;
        let rule_span = self.peek()?.map(|(_, span)| span).unwrap_or(0..0);

        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::RootKw => {
                    self.next()?;
                    attrs.root = Some(self.expect_node_name()?);
                }
                Token::MatchKw => {
                    self.next()?;
                    patterns = self.parse_rule_patterns_until_rparen()?;
                }
                Token::WhenKw => {
                    self.next()?;
                    attrs.predicates = self.parse_predicate_list()?;
                }
                Token::ReplaceKw => {
                    self.next()?;
                    replace = Some(self.parse_constructor()?);
                }
                Token::CoversKw => {
                    self.next()?;
                    attrs.covers = self.parse_cover_list()?;
                }
                Token::CostKw => {
                    self.next()?;
                    attrs.cost = Some(self.expect_int()?);
                }
                Token::PriorityKw => {
                    self.next()?;
                    attrs.priority = Some(self.expect_int()?);
                }
                _ => return self.err_unexpected(span, "valid rewrite-rule field", &tok),
            }
            self.expect(Token::RParen)?;
        }

        if patterns.is_empty() {
            return Err(ParseError::MissingField {
                span: rule_span.into(),
                field: "(match ...)".to_string(),
            });
        }

        let replace = replace.ok_or(ParseError::MissingField {
            span: rule_span.into(),
            field: "(replace ...)".to_string(),
        })?;

        Ok(Def::RewriteRule(RewriteRuleDef {
            attrs,
            patterns,
            replace,
        }))
    }

    fn parse_combine_rule(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::CombineRule)?;

        let mut attrs = RuleAttrs::default();
        let mut patterns = Vec::new();
        let mut match_kind = MatchKind::Single;
        let mut replace = None;
        let rule_span = self.peek()?.map(|(_, span)| span).unwrap_or(0..0);

        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::RootKw => {
                    self.next()?;
                    attrs.root = Some(self.expect_node_name()?);
                }
                Token::MatchKw => {
                    self.next()?;
                    match_kind = MatchKind::Single;
                    patterns = self.parse_rule_patterns_until_rparen()?;
                }
                Token::MatchPairKw => {
                    self.next()?;
                    match_kind = MatchKind::Pair;
                    patterns = self.parse_pattern_group()?;
                }
                Token::WhenKw => {
                    self.next()?;
                    attrs.predicates = self.parse_predicate_list()?;
                }
                Token::ReplaceKw => {
                    self.next()?;
                    replace = Some(self.parse_constructor()?);
                }
                Token::CoversKw => {
                    self.next()?;
                    attrs.covers = self.parse_cover_list()?;
                }
                Token::CostKw => {
                    self.next()?;
                    attrs.cost = Some(self.expect_int()?);
                }
                Token::PriorityKw => {
                    self.next()?;
                    attrs.priority = Some(self.expect_int()?);
                }
                _ => return self.err_unexpected(span, "valid combine-rule field", &tok),
            }
            self.expect(Token::RParen)?;
        }

        if patterns.is_empty() {
            return Err(ParseError::MissingField {
                span: rule_span.into(),
                field: "(match ...) or (match-pair ...)".to_string(),
            });
        }
        let replace = replace.ok_or(ParseError::MissingField {
            span: rule_span.into(),
            field: "(replace ...)".to_string(),
        })?;

        Ok(Def::CombineRule(CombineRuleDef {
            attrs,
            match_kind,
            patterns,
            replace,
        }))
    }

    fn parse_peephole_rule(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::PeepholeRule)?;

        let mut attrs = RuleAttrs::default();
        let mut patterns = Vec::new();
        let mut replace = None;
        let rule_span = self.peek()?.map(|(_, span)| span).unwrap_or(0..0);

        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            self.expect(Token::LParen)?;
            let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            match tok {
                Token::MatchKw => {
                    self.next()?;
                    patterns = self.parse_rule_patterns_until_rparen()?;
                }
                Token::WhenKw => {
                    self.next()?;
                    attrs.predicates = self.parse_predicate_list()?;
                }
                Token::ReplaceKw => {
                    self.next()?;
                    replace = Some(self.parse_constructor()?);
                }
                Token::CoversKw => {
                    self.next()?;
                    attrs.covers = self.parse_cover_list()?;
                }
                Token::CostKw => {
                    self.next()?;
                    attrs.cost = Some(self.expect_int()?);
                }
                Token::PriorityKw => {
                    self.next()?;
                    attrs.priority = Some(self.expect_int()?);
                }
                _ => return self.err_unexpected(span, "valid peephole-rule field", &tok),
            }
            self.expect(Token::RParen)?;
        }

        if patterns.is_empty() {
            return Err(ParseError::MissingField {
                span: rule_span.into(),
                field: "(match ...)".to_string(),
            });
        }
        let replace = replace.ok_or(ParseError::MissingField {
            span: rule_span.into(),
            field: "(replace ...)".to_string(),
        })?;

        Ok(Def::PeepholeRule(PeepholeRuleDef {
            attrs,
            patterns,
            replace,
        }))
    }

    fn parse_pattern_group(&mut self) -> Result<Vec<Pattern>, ParseError> {
        self.expect(Token::LParen)?;
        let mut patterns = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            patterns.push(self.parse_pattern()?);
        }
        self.expect(Token::RParen)?;
        Ok(patterns)
    }

    fn parse_rule_patterns_until_rparen(&mut self) -> Result<Vec<Pattern>, ParseError> {
        let mut patterns = Vec::new();
        while let Some((tok, _)) = self.peek()? {
            if tok == Token::RParen {
                break;
            }
            patterns.push(self.parse_pattern()?);
        }
        Ok(patterns)
    }

    fn parse_predicate_list(&mut self) -> Result<Vec<PredicateExpr>, ParseError> {
        self.expect(Token::LParen)?;
        let mut predicates = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            predicates.push(self.parse_predicate_expr()?);
        }
        self.expect(Token::RParen)?;
        Ok(predicates)
    }

    fn parse_predicate_expr(&mut self) -> Result<PredicateExpr, ParseError> {
        self.expect(Token::LParen)?;
        let name = self.expect_ident()?;
        let mut args = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            args.push(self.parse_predicate_arg()?);
        }
        self.expect(Token::RParen)?;
        Ok(PredicateExpr { name, args })
    }

    fn parse_predicate_arg(&mut self) -> Result<PredicateArg, ParseError> {
        match self.next()? {
            Some((Token::Dollar, _)) => Ok(PredicateArg::Variable(self.expect_ident()?)),
            Some((Token::At, _)) => Ok(PredicateArg::Node(self.expect_ident()?)),
            Some((Token::Int(i), _)) => Ok(PredicateArg::Int(i)),
            Some((Token::Ident(s), _)) => Ok(PredicateArg::Ident(s)),
            Some((t, span)) => self.err_unexpected(span, "predicate arg", &t),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_cover_list(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect(Token::LParen)?;
        let mut covers = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            covers.push(self.expect_node_name()?);
        }
        self.expect(Token::RParen)?;
        Ok(covers)
    }

    fn expect_int(&mut self) -> Result<i64, ParseError> {
        match self.next()? {
            Some((Token::Int(i), _)) => Ok(i),
            Some((t, span)) => self.err_unexpected(span, "integer", &t),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn is_named_pattern_field(name: &str) -> bool {
        name.chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase())
    }

    fn parse_cond_code_ident(name: &str) -> Option<CondCode> {
        match name {
            "E" => Some(CondCode::E),
            "NE" => Some(CondCode::NE),
            "L" => Some(CondCode::L),
            "LE" => Some(CondCode::LE),
            "G" => Some(CondCode::G),
            "GE" => Some(CondCode::GE),
            "B" => Some(CondCode::B),
            "BE" => Some(CondCode::BE),
            "A" => Some(CondCode::A),
            "AE" => Some(CondCode::AE),
            _ => None,
        }
    }

    fn peek_named_pattern_field(&mut self) -> Result<Option<String>, ParseError> {
        Ok(match self.peek()? {
            Some((Token::Ident(name), _)) if Self::is_named_pattern_field(&name) => Some(name),
            Some((Token::Imm, _)) => Some("imm".to_string()),
            Some((Token::Block, _)) => Some("block".to_string()),
            Some((Token::Global, _)) => Some("global".to_string()),
            _ => None,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        match self.next()? {
            Some((Token::Dollar, _)) => {
                let base = Pattern::Variable(self.expect_ident()?);
                self.attach_optional_node_bind(base)
            }
            Some((Token::Ident(s), span)) => {
                if let Some(cc) = Self::parse_cond_code_ident(&s) {
                    let base = Pattern::CondCode(cc);
                    self.attach_optional_node_bind(base)
                } else {
                    Err(ParseError::UnexpectedToken {
                        span: span.into(),
                        expected: "pattern (Variable, Int, CondCode, or LParen)".into(),
                        found: format!("Ident({s:?})"),
                    })
                }
            }
            Some((Token::Int(i), _)) => {
                let base = Pattern::IntConst(i);
                self.attach_optional_node_bind(base)
            }
            Some((Token::LParen, _)) => self.parse_pattern_after_lparen_consumed(),
            Some((t, span)) => Err(ParseError::UnexpectedToken {
                span: span.into(),
                expected: "pattern (Variable, Int, or LParen)".into(),
                found: format!("{:?}", t),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_pattern_after_lparen_consumed(&mut self) -> Result<Pattern, ParseError> {
        let (tok, _span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        let base = match tok {
            Token::And => {
                self.next()?;
                let mut args = Vec::new();
                while self
                    .peek()?
                    .map_or(false, |(t, _)| t != Token::RParen && t != Token::At)
                {
                    args.push(self.parse_pattern()?);
                }
                Pattern::And(args)
            }
            _ => {
                let opcode = self.expect_ident()?;
                if opcode == "schema" {
                    let schema = self.expect_ident()?;
                    let opcode = self.expect_ident()?;
                    let mut args = Vec::new();
                    while self
                        .peek()?
                        .map_or(false, |(t, _)| t != Token::RParen && t != Token::At)
                    {
                        args.push(self.parse_pattern_arg()?);
                    }
                    Pattern::Schema {
                        schema,
                        opcode,
                        args,
                    }
                } else if opcode == "stackslot" {
                    let inner = self.parse_pattern()?;
                    Pattern::StackSlot(Box::new(inner))
                } else if self.peek()?.map_or(true, |(t, _)| t == Token::RParen) {
                    if let Some(cc) = Self::parse_cond_code_ident(&opcode) {
                        Pattern::CondCode(cc)
                    } else {
                        let args = Vec::new();
                        Pattern::Opcode {
                            opcode,
                            ty: None,
                            args,
                        }
                    }
                } else {
                    let mut args = Vec::new();
                    while self
                        .peek()?
                        .map_or(false, |(t, _)| t != Token::RParen && t != Token::At)
                    {
                        args.push(self.parse_pattern_arg()?);
                    }
                    Pattern::Opcode {
                        opcode,
                        ty: None,
                        args,
                    }
                }
            }
        };

        let base = self.attach_optional_node_bind(base)?;
        self.expect(Token::RParen)?;
        Ok(base)
    }

    fn parse_pattern_arg(&mut self) -> Result<PatternArg, ParseError> {
        match self.peek()? {
            Some((Token::LParen, _)) => {
                self.next()?;
                let named = match self.peek_named_pattern_field()? {
                    Some(_) => {
                        let field = self.expect_ident()?;
                        let pattern = self.parse_pattern()?;
                        self.expect(Token::RParen)?;
                        Some(PatternArg::Named {
                            name: field,
                            pattern: Box::new(pattern),
                        })
                    }
                    _ => None,
                };

                match named {
                    Some(arg) => Ok(arg),
                    None => Ok(PatternArg::Positional(
                        self.parse_pattern_after_lparen_consumed()?,
                    )),
                }
            }
            _ => Ok(PatternArg::Positional(self.parse_pattern()?)),
        }
    }

    fn attach_optional_node_bind(&mut self, pattern: Pattern) -> Result<Pattern, ParseError> {
        if let Some((Token::At, _)) = self.peek()? {
            self.next()?;
            let node = self.expect_ident()?;
            Ok(Pattern::NodeBind {
                inner: Box::new(pattern),
                node,
            })
        } else {
            Ok(pattern)
        }
    }

    fn parse_constructor(&mut self) -> Result<Constructor, ParseError> {
        match self.next()? {
            Some((Token::Dollar, _)) => Ok(Constructor::Variable(self.expect_ident()?)),
            Some((Token::Int(i), _)) => Ok(Constructor::Imm(i)),
            Some((Token::LParen, _)) => {
                let opcode = self.expect_ident()?;
                if opcode == "reg" {
                    let name = self.expect_ident()?;
                    self.expect(Token::RParen)?;
                    return Ok(Constructor::Reg(name));
                }
                let mut args = Vec::new();
                while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                    args.push(self.parse_constructor()?);
                }
                self.expect(Token::RParen)?;
                Ok(Constructor::Inst { opcode, args })
            }
            Some((t, span)) => Err(ParseError::UnexpectedToken {
                span: span.into(),
                expected: "constructor (Variable, Int, or LParen)".into(),
                found: format!("{:?}", t),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_def_extractor(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefExtractor)?;
        self.expect(Token::LParen)?;
        let name = self.expect_ident()?;
        let mut args = Vec::new();
        while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
            self.expect(Token::Dollar)?;
            args.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;
        let body = self.parse_pattern()?;
        Ok(Def::Extractor(ExtractorDef { name, args, body }))
    }

    fn parse_def_macro(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefMacro)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        while let Some((t, _)) = self.peek()? {
            if t == Token::RParen {
                break;
            }
            self.expect(Token::Dollar)?;
            args.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;
        let body = self.parse_expr()?;
        Ok(Def::Macro(MacroDef { name, args, body }))
    }

    fn parse_def_template(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefTemplate)?;
        let name = self.expect_ident()?;
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        while let Some((t, _)) = self.peek()? {
            if t == Token::RParen {
                break;
            }
            if let Some((Token::Dollar, _)) = self.peek()? {
                self.next()?;
            }
            args.push(self.expect_ident()?);
        }
        self.expect(Token::RParen)?;

        let mut operands = Vec::new();
        let mut implicit_uses = Vec::new();
        let mut implicit_defs = Vec::new();
        let mut clobbers = Vec::new();
        let mut emit = Vec::new();

        while let Some((Token::LParen, _)) = self.peek()? {
            self.next()?;
            let (tok, _) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            self.parse_inst_body_part(
                tok,
                true,
                &mut operands,
                &mut implicit_uses,
                &mut implicit_defs,
                &mut clobbers,
                &mut emit,
            )?;
            self.expect(Token::RParen)?;
        }

        Ok(Def::Template(TemplateDef {
            name,
            args,
            operands,
            implicit_uses,
            implicit_defs,
            clobbers,
            emit,
        }))
    }

    fn parse_def_inst(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefInst)?;
        let name = self.expect_ident()?;

        let mut template = None;
        let mut operands = Vec::new();
        let mut implicit_uses = Vec::new();
        let mut implicit_defs = Vec::new();
        let mut clobbers = Vec::new();
        let mut emit = Vec::new();

        while let Some((Token::LParen, _)) = self.peek()? {
            self.next()?;
            let (tok, _) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;

            match tok {
                Token::Template => {
                    self.next()?;
                    let t_name = self.expect_ident()?;
                    let mut args = Vec::new();
                    while let Some((t, _)) = self.peek()? {
                        if t == Token::RParen {
                            break;
                        }
                        args.push(self.parse_expr()?);
                    }
                    template = Some(TemplateInst { name: t_name, args });
                }
                _ => {
                    self.parse_inst_body_part(
                        tok,
                        true,
                        &mut operands,
                        &mut implicit_uses,
                        &mut implicit_defs,
                        &mut clobbers,
                        &mut emit,
                    )?;
                }
            }
            self.expect(Token::RParen)?;
        }

        Ok(Def::Inst(InstDef {
            name,
            template,
            operands,
            implicit_uses,
            implicit_defs,
            clobbers,
            emit,
        }))
    }

    fn parse_def_pseudo_inst(&mut self) -> Result<Def, ParseError> {
        self.expect(Token::DefPseudoInst)?;
        let name = self.expect_ident()?;

        let mut operands = Vec::new();
        let mut implicit_uses = Vec::new();
        let mut implicit_defs = Vec::new();
        let mut clobbers = Vec::new();
        let mut emit = Vec::new();

        while let Some((Token::LParen, _)) = self.peek()? {
            self.next()?;
            let (tok, _) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
            self.parse_inst_body_part(
                tok,
                false,
                &mut operands,
                &mut implicit_uses,
                &mut implicit_defs,
                &mut clobbers,
                &mut emit,
            )?;
            self.expect(Token::RParen)?;
        }

        Ok(Def::PseudoInst(PseudoInstDef {
            name,
            operands,
            implicit_uses,
            implicit_defs,
            clobbers,
        }))
    }

    fn parse_inst_body_part(
        &mut self,
        tok: Token,
        allow_encode: bool,
        operands: &mut Vec<OperandConstraint>,
        implicit_uses: &mut Vec<String>,
        implicit_defs: &mut Vec<String>,
        clobbers: &mut Vec<String>,
        emit: &mut Vec<EmitExpr>,
    ) -> Result<(), ParseError> {
        match tok {
            Token::Operands => {
                self.next()?;
                while let Some((Token::LParen, _)) = self.peek()? {
                    operands.push(self.parse_operand_constraint()?);
                }
            }
            Token::ImplicitUses => {
                self.next()?;
                while let Some((Token::Ident(_), _)) = self.peek()? {
                    implicit_uses.push(self.expect_ident()?);
                }
            }
            Token::ImplicitDefs => {
                self.next()?;
                while let Some((Token::Ident(_), _)) = self.peek()? {
                    implicit_defs.push(self.expect_ident()?);
                }
            }
            Token::Clobbers => {
                self.next()?;
                while let Some((Token::Ident(_), _)) = self.peek()? {
                    clobbers.push(self.expect_ident()?);
                }
            }
            Token::Encode if allow_encode => {
                self.next()?;
                while let Some((Token::LParen, _)) = self.peek()? {
                    emit.push(self.parse_emit_expr()?);
                }
            }
            _ => {
                return Err(ParseError::UnexpectedToken {
                    span: self.peeked.as_ref().unwrap().1.clone().into(),
                    expected: if allow_encode {
                        "operands, implicit-uses, implicit-defs, clobbers, or encode".to_string()
                    } else {
                        "operands, implicit-uses, implicit-defs, or clobbers".to_string()
                    },
                    found: token_to_string(&tok),
                });
            }
        }
        Ok(())
    }

    fn parse_operand_constraint(&mut self) -> Result<OperandConstraint, ParseError> {
        self.expect(Token::LParen)?;
        let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        let res = match tok {
            Token::Use => {
                self.next()?;
                match self.peek()? {
                    Some((Token::Dollar, _)) => {
                        self.next()?;
                        OperandConstraint::Use(self.expect_ident()?)
                    }
                    Some((Token::LParen, _)) => {
                        self.next()?;
                        self.expect(Token::Fixed)?;
                        let reg = self.expect_ident()?;
                        if let Some((Token::Dollar, _)) = self.peek()? {
                            self.next()?;
                        }
                        let src = self.expect_ident()?;
                        self.expect(Token::RParen)?;
                        OperandConstraint::FixedUse { reg, src }
                    }
                    Some((t, span)) => return self.err_unexpected(span, "$ or (", &t),
                    None => return Err(ParseError::UnexpectedEof),
                }
            }
            Token::Def => {
                self.next()?;
                match self.peek()? {
                    Some((Token::Dollar, _)) => {
                        self.next()?;
                        OperandConstraint::Def(self.expect_ident()?)
                    }
                    Some((Token::LParen, _)) => {
                        self.next()?;
                        self.expect(Token::Tied)?;
                        if let Some((Token::Dollar, _)) = self.peek()? {
                            self.next()?;
                        }
                        let dst = self.expect_ident()?;
                        if let Some((Token::Dollar, _)) = self.peek()? {
                            self.next()?;
                        }
                        let src = self.expect_ident()?;
                        self.expect(Token::RParen)?;
                        OperandConstraint::TiedDef { dst, src }
                    }
                    Some((t, span)) => return self.err_unexpected(span, "$ or (", &t),
                    None => return Err(ParseError::UnexpectedEof),
                }
            }
            Token::Tied => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                let dst = self.expect_ident()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                let src = self.expect_ident()?;
                OperandConstraint::TiedDef { dst, src }
            }
            Token::Imm => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                OperandConstraint::Imm(self.expect_ident()?)
            }
            Token::Block => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                OperandConstraint::Block(self.expect_ident()?)
            }
            Token::Global => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                OperandConstraint::Global(self.expect_ident()?)
            }
            Token::Ident(name) if name == "stackslot" => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                OperandConstraint::StackSlot(self.expect_ident()?)
            }
            _ => {
                return self.err_unexpected(
                    span,
                    "use, def, fixed, tied, imm, block, global, or stackslot",
                    &tok,
                )
            }
        };
        self.expect(Token::RParen)?;
        Ok(res)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        match tok {
            Token::Dollar => {
                self.next()?;
                Ok(Expr::Variable(self.expect_ident()?))
            }
            Token::Int(n) => {
                self.next()?;
                Ok(Expr::Int(n))
            }
            Token::LParen => {
                self.next()?;
                let name = self.expect_ident()?;
                let mut args = Vec::new();
                while self.peek()?.map_or(false, |(t, _)| t != Token::RParen) {
                    args.push(self.parse_expr()?);
                }
                self.expect(Token::RParen)?;

                match name.as_str() {
                    "hw-enc" => {
                        if args.len() != 1 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "hw-enc".to_string(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        match &args[0] {
                            Expr::Variable(v) => Ok(Expr::HwEnc(v.clone())),
                            _ => Err(ParseError::InvalidArgError {
                                span: span.into(),
                                name: "hw-enc".to_string(),
                                message: "expected a variable for hw-enc".to_string(),
                            }),
                        }
                    }
                    "slot-base-hw-enc" => {
                        if args.len() != 1 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "slot-base-hw-enc".to_string(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        match &args[0] {
                            Expr::Variable(v) => Ok(Expr::SlotBaseHwEnc(v.clone())),
                            _ => Err(ParseError::InvalidArgError {
                                span: span.into(),
                                name: "slot-base-hw-enc".to_string(),
                                message: "expected a stackslot variable".to_string(),
                            }),
                        }
                    }
                    "slot-offset" => {
                        if args.len() != 1 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "slot-offset".to_string(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        match &args[0] {
                            Expr::Variable(v) => Ok(Expr::SlotOffset(v.clone())),
                            _ => Err(ParseError::InvalidArgError {
                                span: span.into(),
                                name: "slot-offset".to_string(),
                                message: "expected a stackslot variable".to_string(),
                            }),
                        }
                    }
                    "slot-size" => {
                        if args.len() != 1 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "slot-size".to_string(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        match &args[0] {
                            Expr::Variable(v) => Ok(Expr::SlotSize(v.clone())),
                            _ => Err(ParseError::InvalidArgError {
                                span: span.into(),
                                name: "slot-size".to_string(),
                                message: "expected a stackslot variable".to_string(),
                            }),
                        }
                    }
                    "slot-align" => {
                        if args.len() != 1 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "slot-align".to_string(),
                                expected: 1,
                                found: args.len(),
                            });
                        }
                        match &args[0] {
                            Expr::Variable(v) => Ok(Expr::SlotAlign(v.clone())),
                            _ => Err(ParseError::InvalidArgError {
                                span: span.into(),
                                name: "slot-align".to_string(),
                                message: "expected a stackslot variable".to_string(),
                            }),
                        }
                    }
                    "bit-or" => {
                        if args.len() != 2 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "bit-or".to_string(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        Ok(Expr::BitOr(
                            Box::new(args[0].clone()),
                            Box::new(args[1].clone()),
                        ))
                    }
                    "bit-and" => {
                        if args.len() != 2 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "bit-and".to_string(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        Ok(Expr::BitAnd(
                            Box::new(args[0].clone()),
                            Box::new(args[1].clone()),
                        ))
                    }
                    "shl" => {
                        if args.len() != 2 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "shl".to_string(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        Ok(Expr::Shl(
                            Box::new(args[0].clone()),
                            Box::new(args[1].clone()),
                        ))
                    }
                    "shr" => {
                        if args.len() != 2 {
                            return Err(ParseError::ArgumentCountError {
                                span: span.into(),
                                name: "shr".to_string(),
                                expected: 2,
                                found: args.len(),
                            });
                        }
                        Ok(Expr::Shr(
                            Box::new(args[0].clone()),
                            Box::new(args[1].clone()),
                        ))
                    }
                    _ => Ok(Expr::Call(name, args)),
                }
            }
            _ => self.err_unexpected(span, "Variable, Int, or LParen", &tok),
        }
    }

    fn parse_emit_expr(&mut self) -> Result<EmitExpr, ParseError> {
        self.expect(Token::LParen)?;
        let (tok, span) = self.peek()?.ok_or(ParseError::UnexpectedEof)?;
        let res = match tok {
            Token::Byte => {
                self.next()?;
                match self.peek()?.ok_or(ParseError::UnexpectedEof)? {
                    (Token::Int(n), _) => {
                        self.next()?;
                        EmitExpr::Byte(n as u8)
                    }
                    _ => EmitExpr::ByteExpr(Box::new(self.parse_expr()?)),
                }
            }
            Token::Imm16 => {
                self.next()?;
                EmitExpr::Imm16(Box::new(self.parse_expr()?))
            }
            Token::Imm32 => {
                self.next()?;
                EmitExpr::Imm32(Box::new(self.parse_expr()?))
            }
            Token::Imm64 => {
                self.next()?;
                EmitExpr::Imm64(Box::new(self.parse_expr()?))
            }
            Token::Rel32 => {
                self.next()?;
                if let Some((Token::Dollar, _)) = self.peek()? {
                    self.next()?;
                }
                EmitExpr::Rel32(self.expect_ident()?)
            }
            Token::If => {
                self.next()?;
                let cond = self.parse_expr()?;
                let mut then_arm = Vec::new();
                while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
                    then_arm.push(self.parse_emit_expr()?);
                }
                let mut else_arm = Vec::new();
                if let Some((Token::Else, _)) = self.peek()? {
                    self.next()?;
                    while self.peek()?.map_or(false, |(t, _)| t == Token::LParen) {
                        else_arm.push(self.parse_emit_expr()?);
                    }
                }
                EmitExpr::If(Box::new(cond), then_arm, else_arm)
            }
            _ => return self.err_unexpected(span, "byte, imm16, imm32, imm64, rel32, or if", &tok),
        };
        self.expect(Token::RParen)?;
        Ok(res)
    }
}

pub fn parse(input: &str) -> Result<Module, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_module()
}
