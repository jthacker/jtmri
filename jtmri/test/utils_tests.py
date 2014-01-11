from ..utils import asiterable

def test_asiterable():
    assert 'x' == asiterable('x')
    assert [1] == list(asiterable(1))
    assert [1,2,3] == list(asiterable([1,2,3]))
