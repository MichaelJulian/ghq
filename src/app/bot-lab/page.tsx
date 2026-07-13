import type { Metadata } from "next";
import BotLab from "@/components/bot-lab/BotLab";

export const metadata: Metadata = {
  title: "GHQ Bot Lab",
  description: "Watch GHQ AI characters play and analyze any GHQ position.",
};

export default function BotLabPage() {
  return <BotLab />;
}
