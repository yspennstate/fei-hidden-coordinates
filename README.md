# Hidden coordinates and selection rules in the Fourier entropy-influence conjecture

Yitzchak Shmalo

This repository contains a paper on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai, together
with the programs behind every numerical value in it.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

For a Boolean function f and a coordinate i, the erasure entropy E_i(f) is the conditional entropy of the event
"i is in U" given U minus i, where U is drawn from the spectral distribution. FEI is equivalent to every non-constant
f having a coordinate with E_i(f) = O(Inf_i(f)).

* The monotone function f_k = MAJ_k(y) OR (x AND [y >= y0]), with y0 of weight (k-1)/2, has E_x/Inf_x tending to
  infinity, yet the sum of all E_i is at most 2 I(f_k). The proof uses that every E_i is at most the binary entropy of
  the Fourier weight on even levels, which is 2^-k here.
* After each y_j is replaced by a self-dual monotone function of fresh variables, E_x is unchanged, every other
  variable has E_l < 4 Inf_l (an exact formula for E_l in terms of the spectrum of f_k), and the sum of the E_i is at
  most 5 I.
* The coordinate of smallest influence, of largest influence, of largest or of smallest normalised level-two weight,
  and the variable lying in the most shortest minterms all fail as selection rules for monotone functions. For the
  smallest level-two weight the hidden coordinate is made pivotal at two complementary middle-layer points.
* An exchangeable law on subsets satisfying the two singleton identities of monotone functions has E_i / Pr[i in U] at
  least log2(n)/288, so those identities alone do not bound the erasure entropy.
* Exact maxima of min_i E_i/Inf_i, H/I and sum_i E_i / I over all monotone functions on at most seven variables.
* The rule that picks, among the variables lying both in a shortest minterm and in a shortest maxterm, one of
  smallest ratio also fails. It is bounded when a shortest minterm or maxterm is a single variable x (then f = x AND g
  up to duality, and E_x/Inf_x depends only on p = Pr[g], is at most 4(1-p) and at most 2.6974), and on all monotone
  functions of at most six variables its worst case equals the best coordinate's. But a variant of f_k puts the hidden
  coordinate in both sets with unbounded ratio, and substituting threshold functions T_t (at least t of 2t) and their
  duals leaves it the only variable in both, with ratio tending to the unsubstituted one as t grows.

## Code

`code/check_hidden.py` (Python 3, NumPy) checks the lemmas, propositions and rule failures on exact Walsh-Hadamard
transforms of f_k for odd k <= 13, of substituted functions, and of the complementary-pivot functions for even
k <= 18; its output is `code/check_hidden_out.txt`.

`code/census_mono.py` (Python 3, NumPy) evaluates every labelled monotone function on n <= 6 variables, with the four
coordinate rules and ties broken against the rule. Outputs: `code/census_n5.json` (n <= 5) and `code/census_n6.json`
(n <= 6). Run `python census_mono.py 6`.

`code/mono_census.c` (C99) runs over orbit representatives of the first n-1 variables and checks the weighted count
against the Dedekind number. Outputs: `code/mono_census_n6.txt` and `code/mono_census_n7.txt` (the n = 7 run evaluated
6,031,580,181 representatives in six parts). Build with `cc -O2 -o mono_census mono_census.c -lm`; usage is in the
header.

`code/rule_minmax.py` (Python 3, NumPy; imports `census_mono.py`) computes, for every monotone function on
n <= 6 variables, the variables in shortest minterms and in shortest maxterms and the best and worst ratios over each
set. Outputs: `code/rule_minmax_n5.json` (n <= 5) and `code/rule_minmax_n6.json` (n = 6).

`code/check_xand.py` checks the formula for x AND g against exact transforms and evaluates it and its bound over p;
`code/check_gk.py` computes the minterm and maxterm sets and every ratio of f_k and of its variant for odd k <= 13.
`code/forced_rule.py` builds the substituted functions, checks that the hidden coordinate is the only variable in
both sets and that its ratio matches the biased expansion of the variant, and tabulates that ratio.
Outputs: `code/check_xand_out.txt`, `code/check_gk_out.txt`, `code/forced_rule_out.txt`.

## Related papers

1. [The one-coordinate step in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)
2. [Good coordinates in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-good-coordinates)
3. [A variance form of Han's bound on the Fourier entropy](https://github.com/yspennstate/fei-variance-han-bound)
4. [Four equivalent forms of the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-equivalent-forms)
5. [A filtration bound for the erasure entropies of Boolean functions](https://github.com/yspennstate/fei-filtration-witness)
6. [Erasure entropies of sparse linear DNFs](https://github.com/yspennstate/fei-sparse-linear-dnf)
7. [The Fourier entropy of linear monotone DNFs on few variables](https://github.com/yspennstate/fei-linear-dnf-census)

## License

MIT, see [LICENSE](LICENSE).
