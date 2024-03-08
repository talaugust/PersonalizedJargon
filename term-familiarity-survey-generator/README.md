# Term Familiarity Survey Generator

This Google Apps Script generates a Google Form for conducting a survey to evaluate researchers' familiarity with terms in scientific abstracts. The script reads data from a Google Sheet and creates a form with multiple sections and questions based on the provided data.

## Example Survey

This is an example survey: [link](https://docs.google.com/forms/d/e/1FAIpQLSfgDmzQTc1VraMwWy0B12BDhCdSnrnbpxA04U5D6SvjA_9_ug/viewform?usp=sf_link)

## Features

- Automatically creates a Google Form titled "Term Familiarity" based on data from a Google Sheet.
- Adds a text item to collect the respondent's Semantic Scholar ID.
- Provides detailed instructions and descriptions for each section of the survey.
- Generates familiarity score questions for each term, both in general and in the context of the provided abstract.
- Allows respondents to select additional information they would want about each term to better understand the abstract.
- Supports formatting of text using bold and italic styles.
- Adds page breaks between each set of questions for better readability.
- Logs the generated form URL for easy access.

## Usage

1. Create a Google Sheet with the required data in the format specified by the script.
2. Open the Google Apps Script editor and copy the provided code.
3. Modify the sheet name ('Sheet1') and column references according to your sheet structure.
4. Run the `createFormFromSheet()` function to generate the Google Form.
5. Access the generated form using the logged form URL.

## Data Format

The script expects the Google Sheet to have the following columns:

- Column L: Title Text
- Column M: Abstract Text
- Column P: Entity Text (terms separated by newline)
- Column Q: Domain Text

Ensure that your sheet follows this structure for the script to function correctly.

## Dependencies

This script requires the following Google Apps Script services:

- SpreadsheetApp: For accessing and reading data from the Google Sheet.
- FormApp: For creating and customizing the Google Form.

Make sure you have the necessary permissions to use these services in your Google Apps Script project.

## Installation

1. Clone the repository or download the script files.
2. Create a new Google Apps Script project.
3. Copy the code from `Code.gs` into the main script file in your project.
4. Set up your Google Sheet with the required data format.
5. Modify the script as needed (sheet name, column references, etc.).
6. Run the `createFormFromSheet()` function to generate the Google Form.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

- The script utilizes the Google Apps Script platform and its services.
- Special thanks to the contributors and developers of the Google Apps Script community.

## Contact

For any questions or inquiries, please contact Yue Guo at yguo50@uw.edu.