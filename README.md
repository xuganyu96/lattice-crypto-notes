# What's next

- [ ] Write a section on GGH trapdoor's application: pkcs and digital signature
- [ ] Write a section on the LLL algorithm itself
- [ ] Read up on "dual lattice"

## References
- Application of Lattice trapdoor
    - [(Peikert, Waters, 08')Application of lossy trapdoor functions](https://eprint.iacr.org/2007/279.pdf)
    - [(Peikert, 09') Public-Key Cryptosystems from the Worst-Case Shortest Vector Problem](https://eprint.iacr.org/2008/481.pdf) (another CCA-secure system, but based on injective trapdoors)
- Lattice trapdoor construction
    - [(GGH, 97') Public-key cryptosystems from lattice reduction problems](https://www.wisdom.weizmann.ac.il/~oded/PSX/pkcs.pdf) generates lattice trapdoor by applying unimodular transformation to short basis to obtain bad basis
    - [(Ajtai, 99') Generating hard instances of the short basis problem](https://people.csail.mit.edu/vinodv/CS294/ajtai99.pdf)
    - [(Micciancio, 01')](https://cseweb.ucsd.edu/~daniele/papers/HNFcrypt.pdf) generate bad basis using Hermite normal form
    - [(Gentry, Peikert, Vaikuntanathan, 08') Trapdoors for hard lattices and new cryptographic constructions](https://eprint.iacr.org/2007/432)
    - [(Alwen, Peiker, 09') Generating shorter bases for hard random lattices](https://eprint.iacr.org/2008/521.pdf)
    - [(Micciancio, Peikert, 12') Trapdoors for lattices: simpler, tighter, faster, smaller](https://eprint.iacr.org/2011/501)
    - [(Micciancio, Genise, 17') Faster Gaussian sampling for trapdoor lattices with arbitrary modulus](https://eprint.iacr.org/2017/308)
    - [MIT CS294 lecture notes](https://people.csail.mit.edu/vinodv/CS294/lecturenotes.pdf)
- Digital signatures
    - [Practical Lattice-based Digital Signature Schemes](https://csrc.nist.gov/csrc/media/events/workshop-on-cybersecurity-in-a-post-quantum-world/documents/papers/session9-oneill-paper.pdf)
    - [(GLP 12') Practical Lattice-Based Cryptography: A Signature Scheme for Embedded Systems](https://www.iacr.org/archive/ches2012/74280529/74280529.pdf)