#include <assert.h>
#include <stdlib.h>
#include "dd_chkpt.h"

int main(void)
{
  assert(get_bar() == BAR);

  return EXIT_SUCCESS;
}
