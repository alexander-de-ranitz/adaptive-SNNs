import equinox as eqx
import jax
import lineax as lx
from jax import numpy as jnp
from jaxtyping import Array, PyTree
from lineax._tags import (
    diagonal_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
)


class DefaultIfNone(lx.AbstractLinearOperator):
    """Lineax operator that returns a default value if the input is None, otherwise applies another operator."""

    default: Array
    else_do: lx.AbstractLinearOperator
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        default: Array,
        else_do: lx.AbstractLinearOperator,
        tags: frozenset[object] = (),
    ):
        if default.shape != else_do.out_structure().shape:
            raise ValueError("Default shape must match else_do output shape.")
        self.default = default
        self.else_do = else_do
        self.tags = frozenset(tags)

    def mv(self, x):
        if x is None:
            return self.default
        else:
            return self.else_do.mv(x)

    def in_structure(self):
        return self.else_do.in_structure()

    def out_structure(self):
        return self.else_do.out_structure()

    def transpose(self):
        return DefaultIfNone(
            self.default.T,
            self.else_do.transpose(),
            transpose_tags(self.tags),
        )

    def as_matrix(self):
        raise NotImplementedError("DefaultIfNone does not support as_matrix()")


class ElementWiseMul(lx.AbstractLinearOperator):
    """Lineax operator for element-wise multiplication with a vector/matrix/tensor.

    Attributes:
        matrix: The matrix to multiply element-wise with.
        tags: A frozenset of tags for the operator.
    """

    matrix: Array
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(self, matrix: Array, tags: frozenset[object] = ()):
        self.matrix = matrix
        self.tags = frozenset(tags)

    def mv(self, x):
        return jnp.multiply(x, self.matrix)

    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=jnp.shape(self.matrix), dtype=self.matrix.dtype
        )

    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=jnp.shape(self.matrix), dtype=self.matrix.dtype
        )

    def transpose(self):
        if symmetric_tag in self.tags:
            return self
        return ElementWiseMul(self.matrix.T, transpose_tags(self.tags))

    def as_matrix(self):
        return self.matrix


class MixedPyTreeOperator(lx.AbstractLinearOperator):
    """A lineax operator to combine lineax operators and arrays in a pytree structure.

    Attributes:
        pytree: A pytree of lineax operators and arrays.
        tags: A frozenset of tags for the operator.
    """

    pytree: PyTree[lx.AbstractLinearOperator | Array]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        pytree: PyTree[lx.AbstractLinearOperator | Array],
        tags: frozenset[object] = (),
    ):
        self.pytree = pytree
        self.tags = frozenset(tags)

    def mv(self, x):
        """Matrix-vector product with a pytree of lineax operators and arrays.

        For each leaf in the pytree, if it is a lineax operator, apply its mv method.
        If it is an array, compute the tensor dot product. An element is considered a leaf
        if it is either a lineax operator or a jax array.
        """

        def prod(op, xi):
            if isinstance(op, lx.AbstractLinearOperator):
                return op.mv(xi)
            else:
                # Same as in diffrax implementation for control term
                return jnp.tensordot(jnp.conj(op), xi, axes=jnp.ndim(xi))

        return jax.tree.map(
            prod,
            self.pytree,
            x,
            is_leaf=lambda node: isinstance(node, lx.AbstractLinearOperator)
            or isinstance(node, jnp.ndarray),
        )

    def in_structure(self):
        def get_in_structure(op):
            if isinstance(op, lx.AbstractLinearOperator):
                return op.in_structure()
            else:
                return jax.ShapeDtypeStruct(shape=op.shape[1], dtype=op.dtype)

        return jax.tree.map(
            get_in_structure,
            self.pytree,
            is_leaf=lambda x: isinstance(x, lx.AbstractLinearOperator)
            or isinstance(x, jnp.ndarray),
        )

    def out_structure(self):
        def get_out_structure(op):
            if isinstance(op, lx.AbstractLinearOperator):
                return op.out_structure()
            else:
                return jax.ShapeDtypeStruct(shape=op.shape[0], dtype=op.dtype)

        return jax.tree.map(
            get_out_structure,
            self.pytree,
            is_leaf=lambda x: isinstance(x, lx.AbstractLinearOperator)
            or isinstance(x, jnp.ndarray),
        )

    def transpose(self):
        return MixedPyTreeOperator(
            jax.tree.map(
                lambda op: op.transpose()
                if isinstance(op, lx.AbstractLinearOperator)
                else jnp.transpose(op),
                self.pytree,
            )
        )

    def as_matrix(self):
        raise NotImplementedError(
            "MixedPyTreeOperator does not (yet) support as_matrix()"
        )


@lx.is_symmetric.register(MixedPyTreeOperator)
@lx.is_symmetric.register(ElementWiseMul)
@lx.is_symmetric.register(DefaultIfNone)
def _(operator):
    return any(
        tag in operator.tags
        for tag in (
            symmetric_tag,
            positive_semidefinite_tag,
            negative_semidefinite_tag,
            diagonal_tag,
        )
    )
