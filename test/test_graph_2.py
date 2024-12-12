def reverse_string(s):
    """Returns the reversed version of the input string."""
    return s[::-1]


def is_palindrome(s):
    """Checks if the input string is a palindrome."""
    s = s.lower().replace(" ", "")  # Normalize the string
    return s == s[::-1]


def count_vowels(s):
    """Counts the number of vowels in the input string."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)


def capitalize_words(s):
    """Capitalizes the first letter of each word in the input string."""
    return " ".join(word.capitalize() for word in s.split())


def char_frequency(s):
    """Returns a dictionary with the frequency of each character in the input string."""
    frequency = {}
    for char in s:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    return frequency
