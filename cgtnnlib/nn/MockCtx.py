class MockCtx:
    saved_tensors = ()

    """
    Mock version of torch context, for debuggning.
    """
    def save_for_backward(
        self,
        *args,
    ):
        self.saved_tensors = args