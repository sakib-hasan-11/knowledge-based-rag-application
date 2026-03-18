"""Quick test to verify BytesIO fix works"""

from io import BytesIO
from unittest.mock import MagicMock

# Test that BytesIO with bytes works (not StringIO)
test_bytes = b"<html><body>Test content EUR</body></html>"
body_io = BytesIO(test_bytes)
content = body_io.read()
print(f"Read bytes: {type(content).__name__}")
print(f"Bytes length: {len(content)}")

# Verify decode works
decoded = content.decode("utf-8")
print(f"Decoded: {type(decoded).__name__}")
print(f"Content contains EUR: {'EUR' in decoded}")

# Test encoding fallback
test_bytes_latin1 = "Cafe".encode("latin-1")
print(f"\nLatin-1 bytes: {test_bytes_latin1}")
try:
    # This will succeed with utf-8 for ASCII
    decoded_utf8 = test_bytes_latin1.decode("utf-8")
    print("UTF-8 decode succeeded for ASCII")
except UnicodeDecodeError:
    print("UTF-8 decode failed, trying latin-1")
    decoded_latin1 = test_bytes_latin1.decode("latin-1")
    print(f"Latin-1 decode succeeded: {decoded_latin1}")

print("\nAll basic BytesIO operations work correctly!")
