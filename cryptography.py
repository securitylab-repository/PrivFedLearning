import numpy as np
import random
import libnum
import sys
import math

class Cryptography():
    class RSA():
        def __init__(self):
            pass

        def rabinMiller(self, n, d):
            """
            Rabin Miller primality test algorithm
                Parameters:
                    n (int): the number to check for primality
                    d (int): odd positive integer
                Returns:
                    True if n is prime
                    False if n is not prime
            """
            a = random.randint(2, (n - 2) - 2)
            x = pow(a, int(d), n) # a^d%n
            if x == 1 or x == n - 1:
                return True

            # square x
            while d != n - 1:
                x = pow(x, 2, n)
                d *= 2

                if x == 1:
                    return False
                elif x == n - 1:
                    return True

            # is not prime
            return False

        def isPrime(self, n):
            """
            Return True if n prime
            fall back to rabinMiller if uncertain
                Parameters:
                    n (int): the number to check for primality
                Returns:
                    True if prime
                    False if not
            """

            # 0, 1, -ve numbers not prime
            if n < 2:
                return False

            # low prime numbers to save time
            lowPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]

            # if in lowPrimes
            if n in lowPrimes:
                return True

            # if low primes divide into n
            for prime in lowPrimes:
                if n % prime == 0:
                    return False

            # find number c such that c * 2 ^ r = n - 1
            c = n - 1 # c even bc n not divisible by 2
            while c % 2 == 0:
                c /= 2 # make c odd

            # prove not prime 128 times
            for i in range(128):
                if not self.rabinMiller(n, c):
                    return False

            return True

        def generateKeys(self, keysize=1024):
            """
            Generates the keys
                Parameters:
                    keysize (int): size of the key in bits
                Returns:
                    e (int): first part of the public key
                    d (int): first part of the private key
                    N (int): product of p and q
            """
            e = d = N = 0

            # get prime nums, p & q
            p = self.generateLargePrime(keysize)
            q = self.generateLargePrime(keysize)

            print(f"p: {p}")
            print(f"q: {q}")

            N = p * q # RSA Modulus
            phiN = (p - 1) * (q - 1) # totient

            # choose e
            # e is coprime with phiN & 1 < e <= phiN
            while True:
                e = random.randrange(2 ** (keysize - 1), 2 ** keysize - 1)
                if (self.isCoPrime(e, phiN)):
                    break

            # choose d
            # d is mod inv of e with respect to phiN, e * d (mod phiN) = 1
            d = self.modularInv(e, phiN)

            return e, d, N

        def generateLargePrime(self, keysize):
            """
            Return random large prime number of keysize bits in size
                Parameters:
                    keysize (int): number of bits
                Returns:
                    num (int): large prime number
            """
            while True:
                num = random.randrange(2 ** (keysize - 1), 2 ** keysize - 1)
                if (self.isPrime(num)):
                    return num

        def isCoPrime(self, p, q):
            """
            Returns if it's coprime
                Parameters:
                    p (int): firt prime number
                    q (int): second prime number
                Returns:
                    True if gcd(p, q) is 1
            """

            return self.gcd(p, q) == 1

        def gcd(self, p, q):
            """
            Euclidean algorithm to find gcd of p and q
                Parameters:
                    p (int): first prime number
                    q (int): second prime number
                Returns:
                    p (int): the gcd
            """

            while q:
                p, q = q, p % q
            return p

        def egcd(self, a, b):
            """
            Extended Euclidean Algorithm
                Parameters:
                    a (int): first number
                    b (int): second number
                Returns:
                    old_r (int): gcd
                    old_s (int): x
                    old_t (int): y
            """
            s = 0; old_s = 1
            t = 1; old_t = 0
            r = b; old_r = a

            while r != 0:
                quotient = old_r // r
                old_r, r = r, old_r - quotient * r
                old_s, s = s, old_s - quotient * s
                old_t, t = t, old_t - quotient * t

            # return gcd, x, y
            return old_r, old_s, old_t

        def modularInv(self, a, b):
            """
            Computes the modular inverse
                Parameters:
                    a (int): first number
                    b (int): second number
                Returns:
                    x (int): the modular inverse
            """
            gcd, x, y = self.egcd(a, b)

            if x < 0:
                x += b

            return x

        def encrypt(self, e, N, msg):
            """
            Encrypts a message
                Parameters:
                    e (int): first element of public key
                    N (int): product of p and q
                    msg (string): message to encrypt
                Returns:
                    cipher (string): ciphertext
            """
            cipher = ""

            for c in msg:
                m = ord(c)
                cipher += str(pow(m, e, N)) + " "

            return cipher

        def decrypt(self, d, N, cipher):
            """
            Decrypts the ciphertext
                Parameters:
                    d (int): first element of the private keyg
                    N (int): product of p and q
                    cipher (string): text do decipher
                Returns:
                    msg (string): decrypted message
            """
            msg = ""

            parts = cipher.split()
            for part in parts:
                if part:
                    c = int(part)
                    msg += chr(pow(c, d, N))

            return msg

        def init_rsa(self, keysize, msg):
            """
            Initialization of keys and encrypt and decrypts a message
                Parameters:
                    keysize (int): size of the key in bits
                    msg (string): message to cipher and decipher
                Returns:
                    enc (string): the ciphertext
                    dec (string): the deciphered text
            """
            e, d, N = self.generateKeys(keysize)
            enc = self.encrypt(e, N, msg)
            dec = self.decrypt(d, N, enc)
            return enc, dec

    class Elgamal():

        def getG(self, p):
            """
            Get a generator of the cyclic group
                Parameters:
                    p (int): a prime number
                Returns:
                    rand (int): the generator
            """
            for x in range (1,p):
                rand = x
                exp = 1
                next = rand % p

                while (next != 1 ):
                    next = (next * rand) % p
                    exp = exp + 1

                if (exp == p - 1):
                    return rand

        def elgamal(self, M, p):
            """
            El Gamal public key encryption method
                Parameters:
                    M (string): the message (m < p)
                    p (int): a prime number
                Returns:
                    void
                    (p, g, y) (tuple): Bob's public key
                    x (int): Bob's private key
                    M (string): the message
                    (a, b) (tuple): the ciphertext
                    message (string): the deciphered text
            """
            if (len(sys.argv)>1):
                    M=int(sys.argv[1])
            if (len(sys.argv)>2):
                    p=int(sys.argv[2])

            g = self.getG(p)
            x = random.randint(1,p)
            k = random.randint(1,p)
            Y = pow(g,x,p)
            a = pow(g,k,p)
            b = (pow(Y,k,p) * M) % p
            message = (b * libnum.invmod(pow(a,x,p), p)) % p


            print("\nBob public key (P,g,Y) = " , p , g , Y)
            print("Bob private key (x) = ", x)

            print("\nMessage = ", M)
            print("\nAlice select a random k = " , k)
            print("Cipher = ", a, b)
            print("Decrypt = ", message)

    class LWE():

        def __init__(self, nvals, B, e, s, M1, M2, q):
            self.nvals = nvals
            self.B = B
            self.e = e
            self.s = s
            self.M1 = M1
            self.M2 = M2
            self.q = q

        def get_uv(self, A, B, M, q):
        	u = 0
        	v = 0
        	sample = random.sample(range(self.nvals - 1), self.nvals // 4)
        	print(sample)

        	for x in range(0, len(sample)):
        		u = u + (A[sample[x]])
        		v = v + B[sample[x]]

        	v = v + math.floor(q / 2) * M
        	return u % q, v % q

        def get_result(self, u1, v1, u2, v2, q):
        	res = ((v1 - self.s * u1) + (v2 - self.s * u2)) % q

        	if (res > q // 2):
        		return 1

        	return 0

        def tobits(self, val):
        	l = [0] * (8)
        	l[0] = val & 0x1
        	l[1] = (val & 0x2) >> 1
        	l[2] = (val & 0x4) >> 2
        	l[3] = (val & 0x8) >> 3
        	return l

        def lwe(self):

            if (len(sys.argv)>1):
            	self.M1=int(sys.argv[1])

            if (len(sys.argv)>2):
            	self.M2=int(sys.argv[2])

            if (len(sys.argv)>3):
            	self.s=int(sys.argv[3])

            if (len(sys.argv)>4):
            	self.q=int(sys.argv[4])

            A = random.sample(range(self.q), self.nvals)

            for x in range(0, len(A)):
            	self.e.append(random.randint(1,3))
            	self.B.append((A[x] * self.s + self.e[x]) % self.q)

            print("\n------Parameters and keys-------")
            print("Value to cipher:\t", self.M1, self.M2)
            print("Public Key (A):\t", A)
            print("Public Key (B):\t", self.B)
            print("Errors (e):\t\t", self.e)
            print("Secret key:\t\t", self.s)
            print("Prime number:\t\t", self.q)
            print("\n------Sampling Process from public key-------")

            bits1 = self.tobits(self.M1)
            bits2 = self.tobits(self.M2)

            print("Bits to be ciphered:", bits1, bits2)

            u1_1, v1_1 = self.get_uv(A, self.B, bits1[0], self.q)
            u2_1, v2_1 = self.get_uv(A, self.B, bits1[1], self.q)
            u3_1, v3_1 = self.get_uv(A, self.B, bits1[2], self.q)
            u4_1, v4_1 = self.get_uv(A, self.B, bits1[3], self.q)

            u1_2, v1_2 = self.get_uv(A, self.B, bits2[0], self.q)
            u2_2, v2_2 = self.get_uv(A, self.B, bits2[1], self.q)
            u3_2, v3_2 = self.get_uv(A, self.B, bits2[2], self.q)
            u4_2, v4_2 = self.get_uv(A, self.B, bits2[3], self.q)

            print("\n------Results                -----------------")
            print("Result bit0 is", self.get_result(u1_1, v1_1, u1_2, v1_2, self.q))
            print("Result bit1 is", self.get_result(u2_1, v2_1, u2_2, v2_2, self.q))
            print("Result bit2 is", self.get_result(u3_1, v3_1, u3_2, v3_2, self.q))
            print("Result bit3 is", self.get_result(u4_1, v4_1, u4_2, v4_2, self.q))
