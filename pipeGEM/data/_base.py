
class BaseData:
    """Abstract base class for all data objects that attach to a metabolic model.

    Subclasses represent different types of biological data (gene expression,
    medium composition, enzyme kinetics, etc.) that can be aligned with a
    COBRA metabolic model.

    Parameters
    ----------
    hook_name : str
        The model attribute name this data hooks into (e.g., ``"genes"``,
        ``"metabolites"``).  Used internally by :class:`pipeGEM.Model` to
        manage attached data objects.

    Attributes
    ----------
    _hook_name : str
        Model attribute this data is associated with.
    _hooked_attr : dict
        Internal cache of attributes generated during alignment.
    """

    def __init__(self, hook_name):
        self._hook_name = hook_name
        self._hooked_attr = {}

    def clean(self):
        """Reset the alignment cache, clearing all hooked attributes."""
        self._hooked_attr = {}

    def align(self, model, **kwargs):
        """Align this data object with a metabolic model.

        Must be implemented by subclasses to map raw data onto model
        components (reactions, metabolites, genes, etc.).

        Parameters
        ----------
        model : cobra.Model or pipeGEM.Model
            The metabolic model to align against.
        **kwargs
            Subclass-specific alignment options.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError()

