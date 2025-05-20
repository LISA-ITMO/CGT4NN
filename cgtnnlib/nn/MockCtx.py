class MockCtx:
    saved_tensors = ()

    """
    Mock version of torch context, for debuggning.
    """

    def save_for_backward(
        self,
        *args,
    ):
        """
        Saves tensors for use during the backward pass.

            Args:
                *args: The tensors to be saved.  These will be available in the
                    `backward()` method via `self.saved_tensors`.

            Returns:
                None
        """
        self.saved_tensors = args
