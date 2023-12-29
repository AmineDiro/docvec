import init, { VecSearch } from "./pkg/doc_wasm.js";

const k = 5;
function cleanString(inputString) {
  // Remove line breaks and collapse consecutive whitespaces
  var cleanedString = inputString.replace(/\s+/g, " ").replace(/\n/g, " ");
  // Remove leading and trailing whitespaces
  cleanedString = cleanedString.trim();
  return cleanedString;
}

function highlightTextElement(divCell, text, opacityValue) {
  var elements = divCell.querySelectorAll("*");
  // Loop over elements and collect text nodes
  var normalizedOpacity = Math.min(Math.max(opacityValue / k, 0), 1);

  elements.forEach(function (element) {
    var textNode = element.textContent;
    textNode = cleanString(textNode);
    if (textNode.includes(text)) {
      console.log("Element matched : ", element);

      element.classList.add("highlight");
      console.log("opacityValue", opacityValue);
      console.log("normal_opacity", normalizedOpacity);
      element.style.opacity = normalizedOpacity;
    }
  });
}
function highlightTextInElement(element, searchString) {
  var nodeIterator = document.createNodeIterator(element, NodeFilter.SHOW_TEXT);
  var currentNode;

  while ((currentNode = nodeIterator.nextNode())) {
    var originalContent = currentNode.nodeValue;
    var textContent = cleanString(originalContent);

    if (textContent.includes(searchString)) {
      console.log("Found : {%s} in node : %s", searchString, textContent);
      // Update node value
      // currentNode.nodeValue = textContent;
      var startIndex = textContent.indexOf(searchString);
      var endIndex = startIndex + searchString.length;

      var range = document.createRange();
      range.setStart(currentNode, startIndex);
      range.setEnd(currentNode, endIndex);

      var span = document.createElement("span");
      span.className = "highlight";
      range.surroundContents(span);
    }
  }
}

function clearBG() {
  var highlightedElements = document.querySelectorAll(".highlight");

  // Loop through each element and remove the "highlight" class
  highlightedElements.forEach(function (element) {
    element.classList.remove("highlight");
    element.style.opacity = 1;
  });
}

(async () => {
  const initResult = await init();
  // NOTE: Testing random inference
  const search_module = await new VecSearch();
  console.log(search_module);

  var elements = document.getElementById("text-contents");
  // Search
  const button = document.getElementById("searchButton");
  button.addEventListener("click", function () {
    clearBG();
    const txt = document.getElementById("searchText").value;
    console.log(`Performing search for: ${txt}`);
    // Get elements
    // Search for nearest
    search_module.search(txt, k).then((search_results) => {
      console.log("VecSearch results:", search_results);
      search_results.forEach((searchText, index) => {
        highlightTextElement(elements, searchText, k - index + 1);
      });
    });
  });
})();
