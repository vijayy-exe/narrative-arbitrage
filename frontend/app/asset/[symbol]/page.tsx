import AssetDashboard from "@/components/AssetDashboard";

export default async function AssetPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = await params;
  return <AssetDashboard symbol={symbol} />;
}
