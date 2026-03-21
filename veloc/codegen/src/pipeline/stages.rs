#[derive(Debug, Clone, Copy, Default)]
pub struct Untyped;

#[derive(Debug, Clone, Copy, Default)]
pub struct RawMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct BlockParamsLowered;

#[derive(Debug, Clone, Copy, Default)]
pub struct LegalizedMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct AbiLowered;

#[derive(Debug, Clone, Copy, Default)]
pub struct BankSelected;

#[derive(Debug, Clone, Copy, Default)]
pub struct PreIselPrepared;

#[derive(Debug, Clone, Copy, Default)]
pub struct SelectedMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct PostIselOptimized;

#[derive(Debug, Clone, Copy, Default)]
pub struct RegAllocated;

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameFinalized;

#[derive(Debug, Clone, Copy, Default)]
pub struct PrologueEpilogueInserted;
