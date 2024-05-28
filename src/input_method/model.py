import jax
import jax.numpy as jnp
import flax.linen as nn
import functools


class NanoLM(nn.Module):
    """NanoLM model."""

    input_vocab_size: int = 26 + 26 * 26
    output_vocab_size: int = 50304
    num_layers: int = 4
    num_heads: int = 4
    head_size: int = 32
    dropout_rate: float = 0.2
    embed_size: int = 128
    block_size: int = 64

    @nn.compact
    def __call__(self, x, training: bool):
        seq_len = x.shape[1]

        x = nn.Embed(self.input_vocab_size, self.embed_size)(x) + nn.Embed(
            self.block_size, self.embed_size
        )(jnp.arange(seq_len))
        for _ in range(self.num_layers):
            x_norm = nn.LayerNorm()(x)
            x = x + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.head_size,
                out_features=self.head_size * self.num_heads,
                dropout_rate=self.dropout_rate,
            )(
                x_norm,
                x_norm,
                mask=jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]))),
                deterministic=not training,
            )

            x = x + nn.Sequential(
                [
                    nn.Dense(4 * self.embed_size),
                    nn.relu,
                    nn.Dropout(self.dropout_rate, deterministic=not training),
                    nn.Dense(self.embed_size),
                ]
            )(nn.LayerNorm()(x))

        x = nn.LayerNorm()(x)
        return nn.Dense(self.output_vocab_size)(x)

    @functools.partial(jax.jit, static_argnames=("self", "length"))
    def generate(self, rng, params, length):
        def _scan_generate(carry, _):
            random_key, context = carry
            logits = self.apply(params, context, training=False)
            rng, rng_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(
                rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
            )
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (rng, context), new_token

        _, new_tokens = jax.lax.scan(
            _scan_generate,
            (rng, jnp.zeros((1, self.block_size), dtype=jnp.int32)),
            (),
            length=length,
        )
        return new_tokens
