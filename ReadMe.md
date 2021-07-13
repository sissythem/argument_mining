# Argument Mining

## Install Ellogon from SVN

1) https://www.ellogon.org/index.php/support/installing-ellogon/install-ellogon-from-svn
2) sudo apt install tcllib

## Install ellogon

1) https://www.ellogon.org/index.php/support/installing-ellogon/install-ellogon-from-svn
2) https://drive.google.com/file/d/186PrPGfz5lmVYcQC9jJw7rZ1sHWbpJrB/view?usp=sharing
3) source path/to/venv/bin/activate
4) inside ellogon folder --> pip3 install -e .
5) deactivate

## Classes in the dataset:

* ADU labels: B-major_claims --> 38, I-major_claims --> 257, B-claim --> 380, I-claim --> 6389, B-premise -->615,
I-premise --> 8906, O --> 19401
  * argumentative tokens: 16585
  * non argumentative tokens: 19401
* relation labels: support --> 781, attack --> 84, other --> 8705
* stance labels: for --> 289, against --> 42


## Deploy token in DebateLab Group

* username: docker
* token: EMtm3PdDnpx8CX8wGs2q