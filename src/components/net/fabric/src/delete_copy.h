/* Intended for use in classes where -Weffc+ ought to recognize that copying is already impossible but does not, e.g.
 *  - class derived from boost:uncopyable
 *  - class has a declared move constructor or move assignment operator
 */
#define DELETE_COPY(X) \
  X(const X &) = delete; \
  X& operator=(const X &) = delete
