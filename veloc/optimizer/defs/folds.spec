// Scalar constant folds use the operation's value semantics. A matching trap keeps the instruction.
fold IAdd;
fold ISub;
fold IMul;
fold INeg;
fold IAddSatS;
fold ISubSatS;
fold IAddWithOverflow;
fold ISubWithOverflow;
fold IMulWithOverflow;
fold IDivS {
    traps = [
        DivisionByZero(bv.eq(rhs, bv.zero())),
        IntegerOverflow(bv.and(bv.eq(lhs, bv.smin()), bv.eq(rhs, bv.ones())))
    ];
}
fold IDivU {
    traps = [DivisionByZero(bv.eq(rhs, bv.zero()))];
}
fold IRemS {
    traps = [DivisionByZero(bv.eq(rhs, bv.zero()))];
}
fold IRemU {
    traps = [DivisionByZero(bv.eq(rhs, bv.zero()))];
}
fold IAnd;
fold IOr;
fold IXor;
fold IShl;
fold IShrS;
fold IShrU;
fold IRotl;
fold IRotr;
fold IClz;
fold ICtz;
fold IPopcnt;
fold IEqz;
fold Icmp;
fold ExtendS;
fold ExtendU;
fold Wrap;
