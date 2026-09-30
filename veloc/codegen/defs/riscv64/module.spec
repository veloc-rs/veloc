import "abi.spec";
import "features.spec";
import "lower.spec";
cpu generic { name = "generic"; features = [I, M, F, D]; }
cpu c908 { name = "c908"; features = [I, M, F, D, Zba, Zbb]; }
