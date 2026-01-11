""" Abstract base class (ABC) for ParMOO embedders. """

from abc import ABC, abstractmethod


class Embedder(ABC):
    """ ABC describing the embedding of design variables.

    This class contains the following methods:
     * ``getLatentDesTols()``
     * ``getFeatureDesTols()``
     * ``getEmbeddingSize()``
     * ``getInputType()``
     * ``getLowerBounds()``
     * ``getUpperBounds()``
     * ``embed(x)``
     * ``embed_grad(dx)``
     * ``extract(x)``

    """

    @abstractmethod
    def __init__(self, settings):
        """ Constructor for the Embedder class.

        Args:
            settings (dict): Contains any variable information that the user
                might need to provide.

        Returns:
            Embedder: A new Embedder object.

        """

    @abstractmethod
    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding.

        """

    @abstractmethod
    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates that this is a discrete variable.

        """

    @abstractmethod
    def getEmbeddingSize(self):
        """ Get the dimension of the latent (embedded) space.

        Returns:
            int: the dimension of the latent space produced.

        """

    @abstractmethod
    def getInputType(self):
        """ Get the input type for this embedder.

        Note: Whatever the input type, the output type must always be a
        ndarray of one or more continuous variables in some range [lb, ub].

        Returns:
            str: A numpy string representation of the input type from the
            feature space.
            Currently supported values are: ["f8", "i4", "a25", or "u25"].

        """

    @abstractmethod
    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded (latent) space.

        Returns:
            ndarray: A 1D array of lower bounds in embedded space whose size
                matches the output of ``getEmbeddingSize()``.

        """

    @abstractmethod
    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded (latent) space.

        Returns:
            ndarray: A 1D array of upper bounds in embedded space whose size
                matches the output of ``getEmbeddingSize()``.

        """

    def embed(self, x):
        """ Embed a design input as an n-dimensional vector for ParMOO.

        Note: For best performance, make sure that jax can jit this method.

        Args:
            x (stype): The value of the design variable to embed, where
                stype matches the numpy-string type specified by
                getInputType().

        Returns:
            ndarray: A 1D array whose size matches the output of
            getEmbeddingSize() containing the embedding of x.

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "embed method yet.")

    def embed_grad(self, dx):
        """ Embed a partial design gradient as a vector for ParMOO.

        Note: If not implemented, ParMOO will still work with gradient-free
        methods, but will not support autograd features.

        For best performance, make sure that jax can jit this method.

        Args:
            dx (float): The partial design gradient to embed.

        Returns:
            numpy.ndarray: A numpy array of length 1 containing a
            rescaling of x

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "embed_grad method yet.")

    def extract(self, x):
        """ Extract a design input from an n-dimensional vector for ParMOO.

        Note: For best performance, make sure that jax can jit this method.

        Args:
            x (ndarray): A 1D array whose size matches the output of
                getEmbeddingSize() containing the embedding of x.

        Returns:
            stype: The value of the design variable to embed, where stype
            matches the numpy-string type specified by getInputType().

        """

        raise NotImplementedError("This Embedder has not implemented an "
                                  "extract method yet.")
