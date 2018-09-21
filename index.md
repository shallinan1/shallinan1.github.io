---
layout: home
title: Home
---

# Welcome!
Welcome to my personal webpage. I am Skyler Hallinan, an undergraduate student at the University of Washington intending to major in Bioengineering and Applied and Computational Math Sciences: Scientific Computing and Numerical Algorithms

# Important Information
My <a href="/assets/official/Hallinan_Skyler_Resume.pdf" target = "_blank"><b>resume</b></a> <br>
My <a href="/assets/official/UWUnofficialTranscript.pdf" target = "_blank"><b>transcript</b></a> <br>

<!-- Trigger/Open The Modal -->
<button id="myBtn">My schedule</button>

<!-- The Modal -->
<div id="myModal" class="modal">

  <!-- Modal content -->
  <div class="modal-content">
    <span class="close">&times;</span>
    <img class="img-responsive" src="/assets/official/autumnschedule.png" alt="Autumn schedule">
  </div>

</div>

<script>
// Get the modal
var modal = document.getElementById('myModal');

// Get the button that opens the modal
var btn = document.getElementById("myBtn");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal 
btn.onclick = function() {
    modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
</script>

