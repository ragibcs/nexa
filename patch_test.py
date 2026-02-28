def test_patch():
    try:
        import sys
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except Exception as e:
        pass
