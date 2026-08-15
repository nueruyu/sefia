from sefia import Policy, policy


policy(Policy())
policy("not a policy")  # pyright: ignore[reportArgumentType]
