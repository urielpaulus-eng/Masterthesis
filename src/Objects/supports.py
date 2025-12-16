# src/Objects/supports.py
# TimberConcreteComposite = tcc
# Support definitions for the TCC element.
# Values are kept simple and will be interpreted later in the solver setup.

support_tcc = {
    "single_span_beam": {
        "type": "single_span_beam"
    },
    "beam_compression_tension": {
        "type": "beam_compression_tension"
    }
}