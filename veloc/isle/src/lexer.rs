use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // 跳过空白符
#[logos(skip(r";[^\n]*", allow_greedy = true))] // 跳过注释
pub enum Token {
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("$")]
    Dollar,
    #[token("@")]
    At,
    #[token("def-macro")]
    DefMacro,
    #[token("def-template")]
    DefTemplate,
    #[token("def-pseudo-inst")]
    DefPseudoInst,
    #[token("def-feature")]
    DefFeature,
    #[token("def-cpu")]
    DefCpu,
    #[token("def-abi")]
    DefAbi,
    #[token("decl")]
    Decl,
    #[token("features")]
    Features,
    #[token("limitations")]
    Limitations,
    #[token("def-inst")]
    DefInst,
    #[token("operands")]
    Operands,
    #[token("implicit-uses")]
    ImplicitUses,
    #[token("implicit-defs")]
    ImplicitDefs,
    #[token("clobbers")]
    Clobbers,
    #[token("encode")]
    Encode,
    #[token("byte")]
    Byte,
    #[token("imm16")]
    Imm16,
    #[token("imm32")]
    Imm32,
    #[token("imm64")]
    Imm64,
    #[token("rel32")]
    Rel32,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("use")]
    Use,
    #[token("def")]
    Def,
    #[token("fixed")]
    Fixed,
    #[token("tied")]
    Tied,
    #[token("imm")]
    Imm,
    #[token("block")]
    Block,
    #[token("global")]
    Global,
    #[token("type")]
    Type,
    #[token("primitive")]
    Primitive,
    #[token("register-class")]
    RegisterClass,
    #[token("def-reg")]
    DefReg,
    #[token("def-regclass")]
    DefRegClass,
    #[token("size")]
    Size,
    #[token("alias")]
    Alias,
    #[token("class")]
    Class,
    #[token("reserved")]
    Reserved,
    #[token("role")]
    Role,
    #[token("hw-enc")]
    HwEnc,
    #[token("select-rule")]
    SelectRule,
    #[token("combine-rule")]
    CombineRule,
    #[token("peephole-rule")]
    PeepholeRule,
    #[token("def-extractor")]
    DefExtractor,
    #[token("template")]
    Template,
    #[token("and")]
    And,
    #[token("root")]
    RootKw,
    #[token("match")]
    MatchKw,
    #[token("match-pair")]
    MatchPairKw,
    #[token("when")]
    WhenKw,
    #[token("replace")]
    ReplaceKw,
    #[token("emit")]
    EmitKw,
    #[token("covers")]
    CoversKw,
    #[token("cost")]
    CostKw,
    #[token("priority")]
    PriorityKw,
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_-]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"[0-9]+-[0-9]+", |lex| lex.slice().to_string())]
    Range(String),

    #[regex(r"0x[0-9a-fA-F]+", |lex| i64::from_str_radix(&lex.slice()[2..], 16).ok())]
    #[regex(r"0b[01]+", |lex| i64::from_str_radix(&lex.slice()[2..], 2).ok())]
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),

    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    String(String),
}
