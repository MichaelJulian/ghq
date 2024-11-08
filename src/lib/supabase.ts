import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  "https://wjucmtrnmjcaatbtktxo.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqdWNtdHJubWpjYWF0YnRrdHhvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzA3MjIxMjMsImV4cCI6MjA0NjI5ODEyM30.QCUf0-2haXEidfeHYLfryqNk6ABtZXs5ItYZCv2yFOs"
);

export async function getUsernames(userIds: string[]): Promise<string[]> {
  const { data, error } = await supabase
    .from("users")
    .select("username")
    .in("id", userIds);

  if (error) {
    throw error;
  }

  return data.map((user) => user.username);
}
