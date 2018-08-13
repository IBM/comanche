#include "wait_poll.h"

#include <gtest/gtest.h>
#include <chrono> /* seconds */
#include <cstddef> /* size_t */

namespace
{
  /* A callback which simply rejects (for requeue) any callback it comes across */
  Component::IFabric_communicator::cb_acceptance reject(void *, ::status_t, std::uint64_t, std::size_t, void *)
  {
    return Component::IFabric_communicator::cb_acceptance::DEFER;
  }
}

void wait_poll(Component::IFabric_communicator &comm_, std::function<void(void *context, ::status_t)> cb_)
{
  std::size_t ct = 0;
  unsigned delay = 0;
  while ( ct == 0 )
  {
    comm_.wait_for_next_completion(std::chrono::seconds(6000));
    /* To test deferral of completions (poll_completions_tentative), call it. */
    ct += comm_.poll_completions_tentative(reject);
    /* deferrals should not count as completions */
    EXPECT_EQ(ct,0);
    /* To test deferral of deferred completions, call it again. */
    ct += comm_.poll_completions_tentative(reject);
    /* deferrals should not count as completions */
    EXPECT_EQ(ct,0);

    ct += comm_.poll_completions(cb_);
    ++delay;
  }
  /* poll_completions does not always get a completion after wait_for_next_completion returns
   * (does it perhaps return when a message begins to appear in the completion queue?)
   * but it should not take more than two trips through the loop to get the completion.
   *
   * The socketss provider, though takes many more.
   */
  EXPECT_LE(delay,200);
  EXPECT_EQ(ct,1);
}
