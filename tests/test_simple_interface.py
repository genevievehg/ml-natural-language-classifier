from simple_interface import interface

def test_interface(monkeypatch, capsys):
    responses = iter(['happy', 'exit'])
    monkeypatch.setattr('builtins.input', lambda: next(responses))
    interface()
    captured = capsys.readouterr()
    assert '**Emotion Classifier**' in captured.out
    assert 'joy' in captured.out