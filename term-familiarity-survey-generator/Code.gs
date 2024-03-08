function createFormFromSheet() {
    const conv = {
    c: function(text, obj) {return text.replace(new RegExp(`[${obj.reduce((s, {r}) => s += r, "")}]`, "g"), e => {
      const t = e.codePointAt(0);
      if ((t >= 48 && t <= 57) || (t >= 65 && t <= 90) || (t >= 97 && t <= 122)) {
        return obj.reduce((s, {r, d}) => {
          if (new RegExp(`[${r}]`).test(e)) s = String.fromCodePoint(e.codePointAt(0) + d);
          return s;
        }, "")
      }
      return e;
    })},
    bold: function(text) {return this.c(text, [{r: "0-9", d: 120734}, {r: "A-Z", d: 120211}, {r: "a-z", d: 120205}])},
    italic: function(text) {return this.c(text, [{r: "A-Z", d: 120263}, {r: "a-z", d: 120257}])},
    boldItalic: function(text) {return this.c(text, [{r: "A-Z", d: 120315}, {r: "a-z", d: 120309}])},
  };

  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Sheet1');
  var form = FormApp.create('Term Familiarity');
  var data = sheet.getDataRange().getValues();

  form.addTextItem()
    .setTitle('What is your Semantic Scholar id? (Numbers only, e.g., 2124928819)')
    .setRequired(true)
    .setHelpText('If you don\'t know your id, please refer to this: https://www.semanticscholar.org/faq#author-id.');

  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getParent().getId());
  var descriptionText = 'We are conducting a study to evaluate researchers\' familiarity with terms in scientific abstracts. \n\n';
  descriptionText += 'Instruction:\n';
  descriptionText += '1. Please read the given list of terms and rate your familiarity for each term. \n\n'
  descriptionText += '2. Please read the provided abstract. For each bolded term, rate your familiarity for the term in the context of the abstract and specify any additional information you would want about that term. Unlike the general familiarity score, this evaluation considers that some terms may have different meanings in various domains or contexts. Below we provide more details about the familiarity rating and options for requesting additional information. \n\n';
  descriptionText += '3. If there is other information you want, you can also select the "other" option and write down the type of information. \n';
  form.addSectionHeaderItem().setHelpText(descriptionText);

  var descriptionText = conv.boldItalic('Familiarity score \n')
  descriptionText += '    Please rate your familiarity with this term on a scale of 1 to 5. Use a rating of 1 to indicate "Not at all Familiar" and a rating of 5 to indicate "Extremely Familiar."  \n\n'
  descriptionText += conv.bold('Not familiar at all') + ': You have never heard of this term.\n';
  descriptionText += conv.bold('Slightly Familiar') + ': You have heard of this term, but do not know much about it.\n';
  descriptionText += conv.bold('Somewhat familiar') + ': You have some basic understanding of what this term is.\n';
  descriptionText += conv.bold('Very Familiar') + ': You have a strong understanding of this term and could explain it to others.\n';
  descriptionText += conv.bold('Extremely Familiar') + ': You have a deep, comprehensive understanding of this term.\n\n';
  descriptionText += conv.boldItalic('Additional information \n')
  descriptionText += '     Choose any additional information you would want about the term in order to better read and understand the abstract. Additional information is broken down into three types: \n\n';
  descriptionText += conv.bold('Definition/explanation') + ': provides key information on the term independent of any context (e.g., a specific scientific abstract). A definition answers the question, "What is/are [term]?" \n';
  descriptionText += conv.bold('Background/motivation') + ': introduces information that is important for understanding the term in the context of the abstract. Background can provide information about how the term relates to overall problem, significance, and motivation of the abstract. \n';
  descriptionText += conv.bold('Example') + ':  offers specific instances that help illustrate the practical application or usage of the term within the abstract.'
  form.addSectionHeaderItem().setHelpText(descriptionText);


  var descriptionText = 'Current study is considered exempt upon University of Washington IRB (ID: MOD00015554). \n\n';
  descriptionText += 'If you have any questions, please reach out to yguo50@uw.edu.';
  
  form.addSectionHeaderItem().setHelpText(descriptionText);
  form.addPageBreakItem();

  var descriptionText = 'Examples of additional information.\n\n'
  descriptionText += 'Example 1:\n\n';
  descriptionText += conv.bold('Term: ECG \n')
  descriptionText += conv.bold('Definition/explanation') + conv.italic(': ECG stands for electrocardiogram, which is way to record the heart\'s electrical activity through repeated cardiac cycles. \n');
  descriptionText += conv.bold('Background/motivation') + conv.italic(': ECGs are inexpensive, non-invasive, widely available, and rapid diagnostic tests that are frequently obtained in many medical settings. Recent work has shown that ECGs can also be used to help predict a range of conditions beyond their primary use of measuring heart activity. \n');
  descriptionText += conv.bold('Example') + conv.italic(': None. \n\n');

  descriptionText += 'Example 2:\n\n';
  descriptionText += conv.bold('Term: Data splitting \n');
  descriptionText += conv.bold('Definition/explanation') + conv.italic(': Data splitting is when data is divided into two or more subsets. \n');
  descriptionText += conv.bold('Background/motivation') + conv.italic(': In settings where researchers are predicting effects from heterogenous data, data is split between a training and hold-out sample. In order to to make conclusions less dependent and a particular data split, researchers often consider multiple sample splits, and aggregate the results. \n');
  descriptionText += conv.bold('Example') + conv.italic(': For instance, it is common for us to divide the data into training, validation, and testing sets using a split ratio of 70%, 20%, and 10%.');

  form.addSectionHeaderItem().setHelpText(descriptionText);
  form.addPageBreakItem();

  for (var i = 1; i < data.length; i++) {
    var id = data[i][0];
    var titleText = data[i][11];
    var abstractText = data[i][12];
    var entityText = data[i][15];
    var domainText = data[i][16];

    // Add section header for each text pair
    var total_number = data.length - 1;
    var sectionTitle = 'Progress: ' + i + ' / ' + total_number;
    form.addSectionHeaderItem().setTitle(sectionTitle);

    var entityLines = entityText.split('\n');

    // 1. familiarity score instruction
    var descriptionText = conv.boldItalic('Familiarity score \n')
    descriptionText += conv.bold('Not familiar at all') + ': You have never heard of this term.\n';
    descriptionText += conv.bold('Slightly Familiar') + ': You have heard of this term, but do not know much about it.\n';
    descriptionText += conv.bold('Somewhat familiar') + ': You have some basic understanding of what this term is.\n';
    descriptionText += conv.bold('Very Familiar') + ': You have a strong understanding of this term and could explain it to others.\n';
    descriptionText += conv.bold('Extremely Familiar') + ': You have a deep, comprehensive understanding of this term.';
    form.addSectionHeaderItem().setHelpText(descriptionText);

    // 2. general familiarity score
    var sectionTitle1 = 'Section 1: General Familiarity Score';
    form.addSectionHeaderItem().setTitle(sectionTitle1);

    form.addGridItem()
      .setTitle('How familiar were you with these terms?')
      .setRows(entityLines)
      .setColumns(['Not familiar at all', 'Slightly familiar', 'Somewhat familiar', 'Very familiar', 'Extremely familiar'])
      .setRequired(true);

    // 3. in-context familiairy score
    var sectionTitle2 = 'Section 2: In-context Familiarity Score';
    form.addSectionHeaderItem().setTitle(sectionTitle2);

    var descriptionText = 'Title: ' + titleText + '\n\n';
    descriptionText += 'Domain: ' + domainText + '\n\n';
    descriptionText += 'Abstract: \n\n';
    entityLines.forEach(function(entity) {
      var re = new RegExp(entity, 'gi'); // 'i' flag for case-insensitive match
      abstractText = abstractText.replace(re, function(match) {
        return conv.bold(match); // Transform the matched substring to bold while preserving its original case
      });
    });
    descriptionText += abstractText;
    form.addSectionHeaderItem().setHelpText(descriptionText);

    form.addGridItem()
      .setTitle('How familiar were you with these terms, ' + conv.bold('IN THE CONTEXT') + ' of this abstract?')
      .setRows(entityLines)
      .setColumns(['Not familiar at all', 'Slightly familiar', 'Somewhat familiar', 'Very familiar', 'Extremely familiar'])
      .setRequired(true);

    // 4. additional information needs
    var sectionTitle3 = 'Section 3: Additional Information Needs';
    form.addSectionHeaderItem().setTitle(sectionTitle3);

    form.addCheckboxGridItem()
      .setTitle('Choose any ' + conv.bold('ADDITIONAL') + ' information not already present in the abstract that you would want about the term in order for ' + conv.bold('YOU') + ' to better read and understand the abstract:')
      .setRows(entityLines)
      .setColumns(['Definition/Explanation', 'Background/Motivation', 'Example', 'Other', 'None'])
      .setRequired(true);
      // .showOtherOption(true);

    form.addTextItem()
      .setTitle('If you chose "Other" as an option above, please provide specific additional information you would like about the term. Please use the following format: [term: additional information;]');

    // Add a page break after each text pair
    if (i != data.length - 1) {
      form.addPageBreakItem();
    }
  }

  Logger.log('Form URL: ' + form.getPublishedUrl());
}

