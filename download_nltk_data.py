import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
resources = ['punkt', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger']
for resource in resources:
    try:
        nltk.download(resource)
        print(f"Successfully downloaded {resource}")
    except Exception as e:
        print(f"Error downloading {resource}: {str(e)}")

print("NLTK data download complete!") 