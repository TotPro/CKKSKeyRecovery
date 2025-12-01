#include <helib/debugging.h>
#include <helib/helib.h>
#include <helib/norms.h>

#include "NTL/GF2E.h"
#include "NTL/GF2X.h"
#include "NTL/ZZ.h"
#include "NTL/ZZX.h"
#include "NTL/ZZ_pE.h"
#include "NTL/ZZ_pX.h"
#include "NTL/tools.h"
#include "eval.h"
#include "helib_utils.h"
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ_pX.h>
#include <iostream>

void NTLpolyQ(NTL::ZZ_pE &pX, const NTL::ZZ *vecZ, const int n) {
  NTL::ZZX zX;
  for (int i = 0; i < n; i++) {
    SetCoeff(zX, i, vecZ[i]);
  }
  pX = NTL::conv<NTL::ZZ_pE>(NTL::conv<NTL::ZZ_pX>(zX));
}

#define Conv2toZ(p) (NTL::conv<NTL::ZZX>(NTL::conv<NTL::GF2X>(p)))
#define ConvZtoQ(p) (NTL::conv<NTL::ZZ_pE>(NTL::conv<NTL::ZZ_pX>(p)))
#define Conv2toQ(p) (ConvZtoQ(Conv2toZ(p)))
#define ConvQtoZ(q) (NTL::conv<NTL::ZZX>(NTL::conv<NTL::ZZ_pX>(q)))
#define ConvZto2(q) (NTL::conv<NTL::GF2E>(NTL::conv<NTL::GF2X>(q)))
#define ConvQto2(q) (ConvZto2(ConvQtoZ(q)))
// void Conv2toZ(NTL::ZZX &out, const NTL::GF2E &in) {
//   NTL::GF2X temp_2x;
//   NTL::conv(temp_2x, in);  // GF2E -> GF2X
//   NTL::conv(out, temp_2x); // GF2X -> ZZX
// }
// void ConvZtoQ(NTL::ZZ_pE &out, const NTL::ZZX &in) {
//   NTL::ZZ_pX temp_px;
//   NTL::conv(temp_px, in);  // ZZX -> ZZ_pX
//   NTL::conv(out, temp_px); // ZZ_pX -> ZZ_pE
// }
// void Conv2toQ(NTL::ZZ_pE &out, const NTL::GF2E &in) {
//   NTL::ZZX temp_z;
//   Conv2toZ(temp_z, in);
//   ConvZtoQ(out, temp_z);
// }
// void ConvQtoZ(NTL::ZZX &out, const NTL::ZZ_pE &in) {
//   NTL::ZZ_pX temp_px;
//   NTL::conv(temp_px, in);  // ZZ_pE -> ZZ_pX
//   NTL::conv(out, temp_px); // ZZ_pX -> ZZX
// }
// void ConvZto2(NTL::GF2E &out, const NTL::ZZX &in) {
//   NTL::GF2X temp_2x;
//   NTL::conv(temp_2x, in);  // ZZX -> GF2X
//   NTL::conv(out, temp_2x); // GF2X -> GF2E
// }
// void ConvQto2(NTL::GF2E &out, const NTL::ZZ_pE &in) {
//   NTL::ZZX temp_z;
//   ConvQtoZ(temp_z, in);
//   ConvZto2(out, temp_z);
// }

// Compute the L-infty norm of aX - bX
// NTL::ZZ maxDiff(NTL::ZZ const *aX, NTL::ZZ const *bX, int n,
//                 NTL::ZZ const &modQ) {
//   NTL::ZZ m = NTL::ZZ::zero();
//   NTL::ZZ hQ = modQ / 2;
//   for (int i = 0; i < n; i++) {
//     NTL::ZZ d = (aX[i] - bX[i]) % modQ;
//     if (d >= hQ) {
//       d = d - modQ;
//     }
//     d = abs(d);
//     if (m < d) {
//       m = d;
//     }
//   }
//   return m;
// }

// Homomorphic computations
helib::Ctxt evalVariance(helib::EncryptedArrayCx const &ea,
                         helib::Ctxt const &ct, size_t n) {
  std::cout << "Compute variance" << std::endl;

  helib::Ctxt ctRes(ct);             // copy x
  ctRes.multiplyBy(ct);              // x^2
  ctRes.dropSmallAndSpecialPrimes(); // drop moduli p_i added in modUp
  for (int i = 2; i <= n; i *= 2) {
    helib::Ctxt tmp(ctRes);
    ea.rotate(tmp, n / i);           // tmp = ctRes >> n/i
    tmp.dropSmallAndSpecialPrimes(); // drop moduli p_i added in modUp

    showCtxtScale(tmp, "rotate ");
    ctRes += tmp;
  }
  ctRes.multByConstantCKKS(1 / (double)n);
  return ctRes;
}

// ctRes[i] = encryption of x^(2^i), where ct = encryption of x, for 0 <= i <=
// logDeg
void evalPowerOf2(std::vector<helib::Ctxt *> &ctRes, helib::Ctxt const &ct,
                  int logDeg) {
  ctRes.resize(logDeg + 1);
  ctRes[0] = new helib::Ctxt(ct); // x^(2^0)
  for (int i = 1; i <= logDeg; i++) {
    ctRes[i] = new helib::Ctxt(*ctRes[i - 1]); // x^(2^{i-1})
    ctRes[i]->multiplyBy(*ctRes[i - 1]);       // x^(2^i)
    ctRes[i]->dropSmallAndSpecialPrimes();
    showCtxtScale(*ctRes[i], "powerOf2 ");
  }
}

// A workaround for multiplying by a constant c, where helib would hit a
// division by 0 error if c<0
void multByConstantCKKSFix(helib::Ctxt &ct, double c) {
  if (c < 0) {
    ct.multByConstantCKKS(-c);
    ct.negate();
  } else {
    ct.multByConstantCKKS(c);
  }
}

// Evaluate a polynomial function up to degree evalDeg
helib::Ctxt evalFunction(helib::Ctxt const &ct, size_t n,
                         std::vector<double> const &coeff, int evalDeg = -1) {
  int deg = evalDeg == -1
                ? coeff.size() - 1
                : std::min((size_t)evalDeg,
                           coeff.size() - 1); // assume coeff is not empty
  int logDeg = std::floor(std::log2((double)deg));
  std::cout << "evalFunction " << coeff << " to degree " << deg << std::endl;
  std::vector<helib::Ctxt *> ctPow2s(logDeg + 1);
  evalPowerOf2(ctPow2s, ct, logDeg);
  helib::Ctxt ctRes(ct);                  // copy x
  multByConstantCKKSFix(ctRes, coeff[1]); // c_1 * x
  ctRes.addConstantCKKS(coeff[0]);        // c_1 * x + c_0
  showCtxtScale(ctRes, "c_1 * x + c_0 ");

  for (int i = 2; i <= deg; i++) {
    if (fabs(coeff[i]) < 1e-27) {
      continue; // Too small, skip this term
    }
    int k = std::floor(std::log2((double)i));
    int r = i - (1 << k);         // i = 2^k + r
    helib::Ctxt tmp(*ctPow2s[k]); // x^(2^k)
    while (r > 0) {
      k = std::floor(std::log2((double)r));
      r = r - (1 << k);
      tmp.multiplyBy(*ctPow2s[k]);
      tmp.dropSmallAndSpecialPrimes();
    }
    multByConstantCKKSFix(tmp, coeff[i]); // c_i * x^i
    showCtxtScale(tmp, "c_i * x^i ");
    ctRes += tmp;
    showCtxtScale(ctRes, "add c_i * x^i ");
  }

  for (auto &x : ctPow2s) {
    delete x;
  }
  return ctRes;
}

void test(int logm, int logp, int logQ, double B, int evalDeg,
          HomomorphicComputation hc) {
  // B is the radius of plaintext numbers
  long m = pow(2, logm); // Zm*
  long r = logp;         // bit precision
  long L = logQ;         // Number of bits of Q

  // Setup the context
  helib::Context context(m, -1, r); // p = -1 => complex field, ie m = p-1

  // context.scale = 10; // used for sampling error bound
  helib::buildModChain(context, L, /*c=*/2); // 2 columns in key switching key
  helib::SecKey secretKey(context);
  secretKey.GenSecKey();

  if (hc == HC_VARIANCE) {
    helib::addSome1DMatrices(
        secretKey); // add rotation keys for variance computation
  }
  helib::PubKey publicKey(secretKey);
  helib::EncryptedArrayCx const &ea(context.ea->getCx());
  long n = ea.size(); // # slots

  ea.getPAlgebra().printout();
  std::cout << "r = " << context.alMod.getR() << std::endl;
  std::cout << "ctxtPrimes=" << context.ctxtPrimes
            << ", ciphertext modulus bits=" << context.bitSizeOfQ() << std::endl
            << std::endl;

#ifdef HELIB_DEBUG
  helib::setupDebugGlobals(&secretKey, context.ea);
#endif

  // Initialize the plaintext vector
  std::vector<std::complex<double>> v1,
      v2; // v1 holds the plaintext input, v2 holds the decryption result
  ea.random(v1, B); // generate a random array of size m/2
  std::cout << "v : size = " << v1.size()
            << ", infty norm = " << largestCxNorm(v1) << std::endl;

  // Encryption
  helib::Ctxt c_v(publicKey); // Ctxt::parts contains the ciphertext polynomials
  ea.encrypt(c_v, publicKey, v1);

  // Homomorphic computation
  std::vector<double> coeff(11);
  helib::Ctxt c_res(publicKey);
  switch (hc) {
  case HC_VARIANCE:
    c_res = evalVariance(ea, c_v, v1.size());
    break;
  case HC_SIGMOID:
    coeff = SpecialFunction::coeffsOf[SpecialFunction::FuncName::SIGMOID];
    c_res =
        evalFunction(c_v, n, coeff, evalDeg); // compute the logistic function
    break;
  case HC_EXP:
    coeff = SpecialFunction::coeffsOf[SpecialFunction::FuncName::EXP];
    c_res = evalFunction(c_v, n, coeff,
                         evalDeg); // compute the exponential function
    break;
  default:
    c_res = c_v; // just copy the input ciphertext
  }
  showCtxtScale(c_res, "result ");
  long logExtraScaling = std::ceil(log2(ea.encodeRoundingError() / 3.5));
  helib::IndexSet s1 = c_res.getPrimeSet();
  while (NTL::log(c_res.getRatFactor()) / log(2.0) > r + logExtraScaling + 10 &&
         s1.card() > 1) {
    s1.remove(s1.last());
    c_res.modDownToSet(s1);
    showCtxtScale(c_res, "modDown");
    s1 = c_res.getPrimeSet();
  }

  // Decryption
  ea.decrypt(c_res, secretKey, v2);

  // Check homomorphic computation error
  std::vector<std::complex<double>> ptRes(v2.size());
  switch (hc) {
  case HC_VARIANCE:
    evalPlainVariance(ptRes, v1);
    break;
  case HC_SIGMOID:
    evalPlainFunc(ptRes, v1, SpecialFunction::SIGMOID, evalDeg);
    break;
  case HC_EXP:
    evalPlainFunc(ptRes, v1, SpecialFunction::EXP, evalDeg);
    break;
  default:
    ptRes = v1;
  }
  std::cout << "computation error = "
            << maxDiff(ptRes, v2) // abs(ptRes[0] - v2[v2.size()-1])
            << ", relative error = " << relError(v2, ptRes)
            << std::endl; // maxDiff(ptRes, v2)/largestElm(ptRes)

  // Key recovery attack ************************************************** //

  // Now let's try to recover sk
  NTL::xdouble scalingFactor = c_res.getRatFactor();

  // Here we use a modified encoding function to round directly into ZZX,
  // instead of rounding to a helib::zzX, which is a vector of long so it could
  // cause integer overflow

  // Recover the encryption error by encoding the approximate plaintext
  // の部分なはず
  NTL::ZZX mPrimeX; // <- error
  helib::CKKS_embedInSlots(mPrimeX, v2, context.zMStar,
                           NTL::to_double(scalingFactor));

  // Check if encoding recovers the decrypted ptxt (before modulo reduction)
  NTL::ZZX encodingDiffX = mPrimeX - helib::decrypted_ptxt_;
  NTL::xdouble mPrimeNorm = helib::coeffsL2Norm(mPrimeX);
  std::cout << "encoding error = " << helib::largestCoeff(encodingDiffX)
            << std::endl;
  std::cout << "m' norm = " << mPrimeNorm
            << ", bits = " << NTL::log(mPrimeNorm) / std::log(2) << std::endl;

  std::cout << "ok" << std::endl;
  NTL::ZZ ql;
  context.productOfPrimes(ql, c_res.getPrimeSet());
  NTL::ZZ_p::init(ql);
  std::cout << "ok1" << std::endl;

  long N = pow(2, logm - 1);
  NTL::ZZX mX = NTL::ZZX(NTL::INIT_MONO, N, 1); // X^N + 1
  NTL::ZZ_pE::init(NTL::conv<NTL::ZZ_pX>(mX));
  std::cout << "ok2" << std::endl;

  NTL::GF2X mX2 = NTL::conv<NTL::GF2X>(mX);
  NTL::GF2E::init(mX2);

  NTL::ZZ_pE aX, eX, sX, bX;

  NTL::ZZX ctxtbX, ctxtaX;
  helib::DoubleCRT ctxtb = c_res.getPart(0);
  helib::DoubleCRT ctxta = c_res.getPart(1);
  ctxtb.toPoly(ctxtbX, true);
  ctxta.toPoly(ctxtaX, true);
  std::cout << "ok3" << std::endl;

  aX = ConvZtoQ(ctxtaX);
  // ConvZtoQ(aX, ctxtaX);
  bX = ConvZtoQ(ctxtbX);
  // ConvZtoQ(bX, ctxtbX);
  eX = ConvZtoQ(mPrimeX);
  // ConvZtoQ(eX, mPrimeX);
  helib::DoubleCRT sk = secretKey.sKeys[0];
  NTL::ZZX skX;
  sk.toPoly(skX, true);
  sX = ConvZtoQ(skX);
  // ConvZtoQ(sX, skX);
  std::cout << "ok4" << std::endl;

  NTL::GF2E aX2 = ConvQto2(aX);
  // NTL::GF2E aX2;
  // ConvQto2(aX2, aX);
  std::cout << "ok5" << std::endl;

  NTL::ZZ_pE s0X = Conv2toQ(ConvQto2(eX - bX) / aX2);
  // NTL::ZZ_pE diff_eb = eX - bX; // 引き算
  // NTL::GF2E diff_eb_2;
  // ConvQto2(diff_eb_2, diff_eb);          // 型変換: ZZ_pE -> GF2E
  // NTL::GF2E term_div1 = diff_eb_2 / aX2; // 割り算
  // NTL::ZZ_pE s0X;
  // Conv2toQ(s0X, term_div1); // 型変換: GF2E -> ZZ_pE
  std::cout << "ok6" << std::endl;
  std::cout << s0X << std::endl;
  NTL::GF2E s2X = ConvQto2(skX);
  std::cout << s2X << std::endl;

  // NTL::ZZX hX = ConvQtoZ(eX - bX + aX * s0X) / 2;
  NTL::ZZX hX = ConvQtoZ(eX - bX + aX * s0X) / 2;
  // NTL::ZZX hX;
  // NTL::ZZX tmp = ConvQtoZ(eX - bX + aX * s0X);
  // NTL::div(hX, tmp, 2);
  // NTL::ZZ_pE inner_term = eX - bX + aX * s0X; // 中身の計算
  // NTL::ZZX inner_term_Z;
  // ConvQtoZ(inner_term_Z, inner_term); // 型変換: ZZ_pE -> ZZX
  // NTL::ZZX hX = inner_term_Z / 2;     // 2で割る
  // NTL::ZZX hX;
  // NTL::ZZX tmp = ConvQtoZ(eX - bX + aX * s0X);
  //
  // // HElib等のRNSベースではパリティが保存されないため、
  // // NTL::div ではなく、係数ごとの切り捨て処理を行う
  // hX = tmp;
  // for (long i = 0; i <= NTL::deg(hX); ++i) {
  //   // 各係数を2で割って切り捨て (floor)
  //   NTL::SetCoeff(hX, i, NTL::coeff(hX, i) / 2);
  // }
  // hX.normalize(); // 正規化（次数の補正）
  std::cout << "ok7" << std::endl;

  NTL::ZZ_pE s1X = Conv2toQ(ConvZto2(hX) / aX2);
  // NTL::GF2E hX_2;
  // ConvZto2(hX_2, hX);               // 型変換: ZZX -> GF2E
  // NTL::GF2E term_div2 = hX_2 / aX2; // 割り算
  // NTL::ZZ_pE s1X;
  // Conv2toQ(s1X, term_div2); // 型変換: GF2E -> ZZ_pE
  std::cout << "ok7" << std::endl;

  // NTL::ZZ_pE ssX = 2 * s1X - s0X;
  NTL::ZZ_pE ssX = 2 * s1X - s0X;

  bool foundKey = (ssX == sX);

  std::cout << (foundKey ? "Found key!" : "Attack failed") << std::endl;
}

int main(int argc, char *argv[]) {
  HomomorphicComputation hc =
      argc > 1 ? parseHC(argv[1]) : HC_NOOP; // Default noop
  long logQ = 300;
  long logm = argc > 2 ? atoi(argv[2]) + 1 : 17;      // m = 2N
  long logp = argc > 3 ? atoi(argv[3]) : 20;          // 20 bit precision
  double plainBound = argc > 4 ? atof(argv[4]) : 1.0; // plaintext size
  long evalDeg = argc > 5 ? atoi(argv[5]) : -1;       // default to all degrees

  std::cout << "Running helib attack for " << hcString(hc) << ", N = 2^"
            << logm - 1 << ", logp = " << logp
            << ", |plaintext| = " << plainBound << ", evalDeg = " << evalDeg
            << std::endl;

  NTL::SetNumThreads(8);
  test(/*logm=*/logm,
       /*logp=*/logp,
       /*logQ=*/logQ,
       /*B=*/plainBound,
       /*evalDeg=*/evalDeg,
       /*hc=*/hc);
  return 0;
}
