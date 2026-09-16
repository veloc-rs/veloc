// These declarations own the runtime representation. Rust implements the
// encoding algorithms, not a second copy of these structs and enums.
type Reg = rust("veloc_encoder::x86_64::Reg");

enum Prefix { variants = [None, P66, F2, F3]; }
enum OpcodeMap { variants = [Primary, Map0F, Map0F38, Map0F3A]; }

struct Legacy {
    prefix: Prefix,
    map: OpcodeMap,
    opcode: u8,
    wide: bool,
}

enum RegField { variants = [Register(Reg), ByteRegister(Reg), Extension(u8)]; }
enum Rm { variants = [Register(Reg), ByteRegister(Reg), Memory(Address)]; }
enum Form { variants = [None, OpcodeReg(Reg), ModRm(RegField, Rm)]; }

enum Scale { variants = [One, Two, Four, Eight]; }
struct Index { reg: Reg, scale: Scale }
struct Memory {
    base: optional(Reg),
    index: optional(Index),
    displacement: i64,
}
// Relative addresses carry only an addend. The caller binds the returned
// fixup to its own symbol/label representation.
enum Address { variants = [BaseIndex(Memory), RipRelative(i64)]; }
enum Immediate {
    variants = [None, Bits8(i64), Bits32(i64), Signed32(i64), Bits64(i64), Relative(i64)];
}

struct Branch {
    map: OpcodeMap,
    near: u8,
    short: u8,
}
