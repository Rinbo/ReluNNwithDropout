## Copyright (C) 2017 robin

## Author: robin <robin@XPS15>
## Created: 2017-06-06

function acc = performance (H, Y)

  [dummy, predicted_digit] = max(H);
  acc = mean(predicted_digit' == Y);

endfunction
