import init, { Embedder } from "./pkg/doc_wasm.js";

function cleanString(inputString) {
  // Remove line breaks and collapse consecutive whitespaces
  var cleanedString = inputString.replace(/\s+/g, " ").replace(/\n/g, " ");
  // Remove leading and trailing whitespaces
  cleanedString = cleanedString.trim();
  return cleanedString;
}
(async () => {
  const initResult = await init();
  // NOTE: Testing random inference
  //   const embeddr = await new Embedder();
  //   console.log(embeddr);
  //   var random_embedding = await embeddr.random_inference();
  //   console.log(random_embedding);

  // Search
  const button = document.getElementById("searchButton");
  button.addEventListener("click", function () {
    const txt = document.getElementById("searchText").value;
    console.log(`Performing search for: ${txt}`);

    // Try highlighting and matching here
    const hlText = "active Python core developers elected ";
    var elements = document
      .getElementById("text-contents")
      .getElementsByTagName("*");

    for (var i = 0; i < elements.length; i++) {
      var element = elements[i];
      var cleanedText = cleanString(element.textContent);
      if (cleanedText.includes(hlText)) {
        var regex = new RegExp("(" + hlText + ")", "gi");
        var highlightedHTML = cleanedText.replace(
          regex,
          '<span class="highlight">$1</span>'
        );
        element.innerHTML = highlightedHTML;
      }
    }
  });
})();
