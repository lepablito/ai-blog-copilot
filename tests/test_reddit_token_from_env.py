from radar.tools.reddit import token_from_env


def test_no_credentials_means_no_token():
    assert token_from_env({}) is None


def test_partial_credentials_mean_no_token():
    """Half a credential pair is a configuration mistake, not an attempt worth
    making — asking Reddit would just spend a round trip on a guaranteed 401."""
    assert token_from_env({"REDDIT_CLIENT_ID": "id"}) is None
    assert token_from_env({"REDDIT_CLIENT_SECRET": "secret"}) is None


def test_credentials_are_exchanged_for_a_token():
    calls = []

    def fake_get_token(client_id, client_secret):
        calls.append((client_id, client_secret))
        return "tok"

    env = {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "secret"}

    assert token_from_env(env, get_token=fake_get_token) == "tok"
    assert calls == [("id", "secret")]


def test_a_failing_token_exchange_degrades_instead_of_raising():
    """Bad credentials must cost the score column, not the whole radar run."""
    from radar.errors import ToolError

    def boom(_id, _secret):
        raise ToolError("HTTP 401")

    env = {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "wrong"}

    assert token_from_env(env, get_token=boom) is None
