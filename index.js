import init, { VecSearch } from "./pkg/doc_wasm.js";

function cleanString(inputString) {
  // Remove line breaks and collapse consecutive whitespaces
  var cleanedString = inputString.replace(/\s+/g, " ").replace(/\n/g, " ");
  // Remove leading and trailing whitespaces
  cleanedString = cleanedString.trim();
  return cleanedString;
}

function highlightTextInElement(element, searchString) {
  try {
    var originalContent = element.textContent;
    var textContent = cleanString(element.textContent);

    if (textContent.includes(searchString)) {
      console.log("FOUND IT", searchString);
      console.log("Element", element);

      var startIndex = textContent.indexOf(searchString);
      var endIndex = startIndex + searchString.length;

      console.log(startIndex, endIndex);
      var childNodes = element.childNodes;
      console.log(childNodes);

      // var range = document.createRange();
      // range.setStart(element.firstChild, startIndex);
      // range.setEnd(element.firstChild, endIndex);

      // var span = document.createElement("span");
      // span.className = "highlight";
      // range.surroundContents(span);
      element.classList.add("highlight");
    }
  } catch (e) {
    console.error(e);
  }
}

function clearBG() {
  var highlightedElements = document.querySelectorAll(".highlight");

  // Loop through each element and remove the "highlight" class
  highlightedElements.forEach(function (element) {
    element.classList.remove("highlight");
  });
}

(async () => {
  const initResult = await init();
  // NOTE: Testing random inference
  const search_module = await new VecSearch();
  console.log(search_module);

  var elements = document
    .getElementById("text-contents")
    .getElementsByTagName("*");
  // Search
  const button = document.getElementById("searchButton");
  button.addEventListener("click", function () {
    clearBG();
    const txt = document.getElementById("searchText").value;
    console.log(`Performing search for: ${txt}`);
    // Get elements
    // Search for nearest
    search_module.search(txt, 5).then((search_results) =>
      search_results.forEach(function (hlText) {
        for (var i = 0; i < elements.length; i++) {
          var element = elements[i];
          highlightTextInElement(element, hlText);
        }
        // Try highlighting and matching here
        // for (var i = 0; i < elements.length; i++) {
        //   var element = elements[i];
        //   var cleanedText = cleanString(element.textContent);
        //   if (cleanedText.includes(hlText)) {
        //     var regex = new RegExp("(" + hlText + ")", "gi");
        //     var highlightedHTML = cleanedText.replace(
        //       regex,
        //       '<span class="highlight">$1</span>'
        //     );
        //     element.innerHTML = highlightedHTML;
        //   }
        // }
      })
    );
  });
})();
