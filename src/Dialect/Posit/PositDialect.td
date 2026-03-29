
#ifndef POSIT_DIALECT
#define POSIT_DIALECT

include "mlir/IR/OpBase.td"


def Posit_Dialect : Dialect {
    let name = "posit"; // MLIR : `!posit.*` 後續Type定義
    let cppNamespace = "::mlir::posit"; // C++ : mlir::posit
    let summary = "Dialect for Posit numeric type";
    let description = [{
        Test create a Dialect.
    }];
    let useDefaultTypePrinterParser = 1; //parse
    let hasConstantMaterializer = 1; // op
}

#endif // POSIT_DIALECT