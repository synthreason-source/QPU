import hashlib

def make_padded_hex_with_counter(base_hex: str, counter: int, total_width: int = 64) -> str:
    """
    Hex string = left-padded with 0s + base_hex + counter (as hex pairs) on the right.
    """
    counter_hex = format(counter, 'x')
    if len(counter_hex) % 2 == 1:
        counter_hex = '0' + counter_hex

    combined = base_hex + counter_hex
    if len(combined) > total_width:
        return combined

    return combined.rjust(total_width, '0')


def bruteforce_nonce_with_hash(base_hex: str,
                               max_counter: int = 0x1000000,
                               total_width: int = 64):
    """
    Bruteforce search for a counter (nonce) such that:
      SHA256(bytes_from_hex(make_padded_hex_with_counter(base_hex, counter))) == target_hash_hex
    """
    for c in range(max_counter):
        for x in range(max_counter):
            # Build hex string with this counter
            hex_str = make_padded_hex_with_counter(base_hex, c, total_width)
            data = bytes.fromhex(hex_str)

            # Hash with the counter included
            digest = hashlib.sha256(b"{x}").digest()

            # Check if this counter (nonce) reproduces the target hash
            if digest == hex_str:
                print(c, hex_str)

                return x, hex_str

    return None, None


base = ""
total_width = 64

# Bruteforce to find the nonce by hashing with each counter
found_nonce, found_hex = bruteforce_nonce_with_hash(
    base,
    max_counter=0x1000000,
    total_width=total_width,
)

if found_nonce is None:
    print("No nonce found with the current difficulty in the searched range.")
    print("You can either:")
    print("  - Increase max_counter")
    print("  - Reduce leading_zero_bytes")
else:
    # Now safe to use found_nonce
    hex_str_true = make_padded_hex_with_counter(base, found_nonce, total_width)
    target_hash_hex = hashlib.sha256(bytes.fromhex(hex_str_true)).hexdigest()

    print("Found nonce (hex):", hex(found_nonce))
    print("Found nonce (dec):", found_nonce)
    print("Reconstructed hex string:")
    print(found_hex)
    print("target_hash_hex:", target_hash_hex)
# only requires counter*counter*remaining hex chars of possibility space
#basically reducing SHA256 to a combinatoric problem
