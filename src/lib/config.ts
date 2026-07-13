export interface Config {
  useClerk: boolean;
}

export const config: Config = {
  // A fresh local/Vercel dashboard deployment may not have the original
  // playghq.com Clerk application configured. In that case, run the public
  // frontend and Bot Lab instead of crashing every route in middleware.
  useClerk:
    process.env.NEXT_PUBLIC_GHQ_FRONTEND_ONLY !== "true" &&
    Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY),
};
