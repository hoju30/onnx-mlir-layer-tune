
#ifndef HELLO_TYPES
#define HELLO_TYPES

include "mlir/IR/AttrTypeBase.td"
include "PositDialect.td"


def Posit_Type : TypeDef<Posit_Dialect, "Posit"> {
    let mnemonic = "type"; 


    let summary = "Parametric Posit number type";
    let description = [{
        Test create a data type.
    }];


    let parameters = (ins
        "unsigned":$nbits,
        "unsigned":$es
    );

    let assemblyFormat = "`<` $nbits `,` $es `>`";

    let genVerifyDecl = 1;
}

#endif // HELLO_TYPES