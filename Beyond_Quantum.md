## Inital process
The first thing I did was read the description and do a quick google on NTRU.
This led me to the wikipedia page, which describes the nature of the algorithm and potential attacks.
Then I downloaded the python files and began looking into how it works.
Finally, I accessed the netcat to verify how it works.

## Next steps: Research
My next step was to research how NTRU works and identify my best avenue for attack. 
An important detail was this challenge was not one that was listed as one that we could brute force, which meant that key search and meet in the middle attacks were off the table.

## Finding the solution
Eventually, I noticed something odd. In my IDE, the second argument in the encrypt function is grayed out, indicating that it isn't used:
```
    def encrypt(self, msg_poly, rand_poly):
        return (((self.h_poly).trunc(self.q) + msg_poly) % self.R_poly).trunc(self.q)
```
Sure enough, it isn't using the randomly generated polynomial. 
According to my research, proper encryption will multiply the public key polynomial (h_poly) by a random polynomial before adding the message.
Without it, decrypting is as simple as subtracting the public key back out and doing the modulo again.
```
def decrypt(h_poly, enc_poly):
    q = 128
    N = 97
    R_poly = Poly(x ** N - 1, x).set_domain(ZZ)
    return ((enc_poly).trunc(q) - (h_poly).trunc(q) % R_poly).trunc(q)
```
(Note N, q, and R_poly are always the same in this challenge)
Finally, I connect, get the polynomial representation of the ciphertext and public keys, copy them into a python IDE, load up my decrypt function, and run it.
Running their `poly_to_bytes` function reveals the password, I enter the password, and capture the flag.
