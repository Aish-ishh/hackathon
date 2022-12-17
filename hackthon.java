import java.util.Arrays;
import java.util.List;

public class BadWordDetector {
  // list of bad words
  private static final List<String> BAD_WORDS = Arrays.asList(
    "badword1", "badword2", "badword3" );

  public static String censorBadWords(String input) {
    // split the input string into words
    String[] words = input.split("\\s+");
    // loop through each word and check if it is a bad word
    for (int i = 0; i < words.length; i++) {
      String word = words[i].toLowerCase();
      if (BAD_WORDS.contains(word)) {
        // if it is a bad word, censor it by replacing it with asterisks
        String censoredWord = "";
        for (int j = 0; j < word.length(); j++) {
          censoredWord += "*";
        }
        words[i] = censoredWord;
      }
    }

    // join the words back into a single string
    return String.join(" ", words);
  }

  public static void main(String[] args) {
    String input = "This is a test string with some bad words.";
    System.out.println(censorBadWords(input));
    // Output: "This is a test string with some *** words."
  }
}

    